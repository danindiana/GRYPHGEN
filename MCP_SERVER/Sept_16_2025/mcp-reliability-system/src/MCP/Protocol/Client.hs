
{-# LANGUAGE OverloadedStrings #-}

module MCP.Protocol.Client
  ( -- * MCP Client
    MCPClient
  , createClient
  , connectClient
  , disconnectClient
  , sendRequest
  , sendNotification
  
    -- * Tool Operations
  , listTools
  , callTool
  , getToolSchema
  
    -- * Resource Operations
  , listResources
  , readResource
  
    -- * Client State
  , getClientState
  , ClientError(..)
  ) where

import Control.Concurrent.STM
import Control.Exception (Exception, try)
import Control.Monad (when)
import Data.Aeson (Value(..), object, (.=))
import Data.Text (Text)
import qualified Data.Text as T
import Data.Time
import Data.UUID (UUID)
import qualified Data.UUID.V4 as UUID

import MCP.Protocol.Types
import MCP.Protocol.JsonRPC

-- | MCP Client implementation
data MCPClient = MCPClient
  { clientId :: !UUID
  , clientConfig :: !TransportConfig
  , clientState :: !(TVar ConnectionState)
  , clientCapabilities :: !(TVar (Maybe MCPCapabilities))
  , clientTools :: !(TVar [MCPTool])
  , clientResources :: !(TVar [MCPResource])
  }

-- | Client errors
data ClientError
  = NotConnected
  | InvalidResponse Text
  | ServerError Text
  | TransportError Text
  deriving (Show, Eq)

instance Exception ClientError

-- | Create a new MCP client
createClient :: TransportConfig -> IO MCPClient
createClient config = do
  clientId <- UUID.nextRandom
  state <- newTVarIO Disconnected
  capabilities <- newTVarIO Nothing
  tools <- newTVarIO []
  resources <- newTVarIO []
  
  return $ MCPClient
    { clientId = clientId
    , clientConfig = config
    , clientState = state
    , clientCapabilities = capabilities
    , clientTools = tools
    , clientResources = resources
    }

-- | Connect to MCP server
connectClient :: MCPClient -> IO (Either ClientError ())
connectClient client = do
  atomically $ writeTVar (clientState client) Connecting
  
  -- Perform connection based on transport type
  result <- case transportType (clientConfig client) of
    StdioTransport -> connectStdio client
    HttpTransport -> connectHttp client
    WebSocketTransport -> connectWebSocket client
  
  case result of
    Left err -> do
      atomically $ writeTVar (clientState client) (Error $ T.pack $ show err)
      return $ Left err
    Right _ -> do
      -- Perform initialization handshake
      initResult <- initializeConnection client
      case initResult of
        Left err -> do
          atomically $ writeTVar (clientState client) (Error $ T.pack $ show err)
          return $ Left err
        Right _ -> do
          atomically $ writeTVar (clientState client) Connected
          return $ Right ()

-- | Disconnect from MCP server
disconnectClient :: MCPClient -> IO ()
disconnectClient client = do
  atomically $ writeTVar (clientState client) Disconnected
  -- TODO: Close actual connection

-- | Send a JSON-RPC request
sendRequest :: MCPClient -> Text -> Maybe Value -> IO (Either ClientError Value)
sendRequest client method params = do
  state <- readTVarIO (clientState client)
  case state of
    Connected -> do
      requestId <- UUID.nextRandom
      let request = createRequest (String $ T.pack $ show requestId) method params
      
      -- Send request and wait for response
      result <- sendJsonRpcRequest client request
      case result of
        Left err -> return $ Left err
        Right response -> 
          case rpcResult response of
            Just result -> return $ Right result
            Nothing -> case rpcError response of
              Just err -> return $ Left $ ServerError $ rpcErrorMessage err
              Nothing -> return $ Left $ InvalidResponse "No result or error in response"
    _ -> return $ Left NotConnected

-- | Send a JSON-RPC notification
sendNotification :: MCPClient -> Text -> Maybe Value -> IO (Either ClientError ())
sendNotification client method params = do
  state <- readTVarIO (clientState client)
  case state of
    Connected -> do
      let notification = createNotification method params
      sendJsonRpcNotification client notification
    _ -> return $ Left NotConnected

-- | List available tools
listTools :: MCPClient -> IO (Either ClientError [MCPTool])
listTools client = do
  result <- sendRequest client "tools/list" Nothing
  case result of
    Left err -> return $ Left err
    Right value -> do
      -- Parse tools from response
      case parseToolsResponse value of
        Left parseErr -> return $ Left $ InvalidResponse parseErr
        Right tools -> do
          atomically $ writeTVar (clientTools client) tools
          return $ Right tools

-- | Call a tool
callTool :: MCPClient -> Text -> Value -> IO (Either ClientError Value)
callTool client toolName arguments = do
  let params = object
        [ "name" .= toolName
        , "arguments" .= arguments
        ]
  sendRequest client "tools/call" (Just params)

-- | Get tool schema
getToolSchema :: MCPClient -> Text -> IO (Either ClientError Value)
getToolSchema client toolName = do
  tools <- readTVarIO (clientTools client)
  case filter (\tool -> toolName == toolName tool) tools of
    [] -> return $ Left $ ServerError "Tool not found"
    (tool:_) -> return $ Right $ toolInputSchema tool

-- | List available resources
listResources :: MCPClient -> IO (Either ClientError [MCPResource])
listResources client = do
  result <- sendRequest client "resources/list" Nothing
  case result of
    Left err -> return $ Left err
    Right value -> do
      case parseResourcesResponse value of
        Left parseErr -> return $ Left $ InvalidResponse parseErr
        Right resources -> do
          atomically $ writeTVar (clientResources client) resources
          return $ Right resources

-- | Read a resource
readResource :: MCPClient -> Text -> IO (Either ClientError Value)
readResource client resourceUri = do
  let params = object ["uri" .= resourceUri]
  sendRequest client "resources/read" (Just params)

-- | Get current client state
getClientState :: MCPClient -> IO ConnectionState
getClientState client = readTVarIO (clientState client)

-- Transport-specific connection implementations
connectStdio :: MCPClient -> IO (Either ClientError ())
connectStdio _client = do
  -- TODO: Implement STDIO transport
  return $ Right ()

connectHttp :: MCPClient -> IO (Either ClientError ())
connectHttp _client = do
  -- TODO: Implement HTTP transport
  return $ Right ()

connectWebSocket :: MCPClient -> IO (Either ClientError ())
connectWebSocket _client = do
  -- TODO: Implement WebSocket transport
  return $ Right ()

-- | Initialize connection with handshake
initializeConnection :: MCPClient -> IO (Either ClientError ())
initializeConnection client = do
  let initParams = object
        [ "protocolVersion" .= ("2025-03-26" :: Text)
        , "capabilities" .= object
            [ "tools" .= True
            , "resources" .= True
            , "prompts" .= False
            , "sampling" .= False
            ]
        , "clientInfo" .= object
            [ "name" .= ("mcp-reliability-client" :: Text)
            , "version" .= ("0.1.0" :: Text)
            ]
        ]
  
  result <- sendRequest client "initialize" (Just initParams)
  case result of
    Left err -> return $ Left err
    Right response -> do
      -- Parse server capabilities
      case parseInitializeResponse response of
        Left parseErr -> return $ Left $ InvalidResponse parseErr
        Right capabilities -> do
          atomically $ writeTVar (clientCapabilities client) (Just capabilities)
          return $ Right ()

-- JSON-RPC transport (placeholder implementations)
sendJsonRpcRequest :: MCPClient -> JsonRpcRequest -> IO (Either ClientError JsonRpcResponse)
sendJsonRpcRequest _client _request = do
  -- TODO: Implement actual JSON-RPC transport
  return $ Right $ createResponse (String "test") (String "success")

sendJsonRpcNotification :: MCPClient -> JsonRpcNotification -> IO (Either ClientError ())
sendJsonRpcNotification _client _notification = do
  -- TODO: Implement actual JSON-RPC transport
  return $ Right ()

-- Response parsing helpers
parseToolsResponse :: Value -> Either Text [MCPTool]
parseToolsResponse value = do
  case fromJSON value of
    Success tools -> Right tools
    Error err -> Left $ T.pack err

parseResourcesResponse :: Value -> Either Text [MCPResource]
parseResourcesResponse value = do
  case fromJSON value of
    Success resources -> Right resources
    Error err -> Left $ T.pack err

parseInitializeResponse :: Value -> Either Text MCPCapabilities
parseInitializeResponse value = do
  case fromJSON value of
    Success capabilities -> Right capabilities
    Error err -> Left $ T.pack err

-- Import for JSON parsing
import Data.Aeson (fromJSON, Result(..))

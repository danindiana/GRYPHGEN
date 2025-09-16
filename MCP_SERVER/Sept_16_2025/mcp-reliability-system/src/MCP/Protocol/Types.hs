
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module MCP.Protocol.Types
  ( -- * JSON-RPC Types
    JsonRpcRequest(..)
  , JsonRpcResponse(..)
  , JsonRpcError(..)
  , JsonRpcNotification(..)
  
    -- * MCP Protocol Types
  , MCPMessage(..)
  , MCPCapabilities(..)
  , MCPTool(..)
  , MCPResource(..)
  , MCPPrompt(..)
  
    -- * Transport Types
  , TransportType(..)
  , TransportConfig(..)
  , ConnectionState(..)
  
    -- * Server Discovery Types
  , ServerInfo(..)
  , ServerCapabilities(..)
  , DiscoveryConfig(..)
  ) where

import Data.Aeson
import Data.Text (Text)
import Data.Time (UTCTime)
import GHC.Generics
import Data.UUID (UUID)

-- | JSON-RPC 2.0 Request
data JsonRpcRequest = JsonRpcRequest
  { rpcId :: !(Maybe Value)
  , rpcMethod :: !Text
  , rpcParams :: !(Maybe Value)
  , rpcJsonrpc :: !Text
  } deriving (Show, Generic)

-- | JSON-RPC 2.0 Response
data JsonRpcResponse = JsonRpcResponse
  { rpcResponseId :: !Value
  , rpcResult :: !(Maybe Value)
  , rpcError :: !(Maybe JsonRpcError)
  , rpcResponseJsonrpc :: !Text
  } deriving (Show, Generic)

-- | JSON-RPC 2.0 Error
data JsonRpcError = JsonRpcError
  { rpcErrorCode :: !Int
  , rpcErrorMessage :: !Text
  , rpcErrorData :: !(Maybe Value)
  } deriving (Show, Generic)

-- | JSON-RPC 2.0 Notification
data JsonRpcNotification = JsonRpcNotification
  { rpcNotificationMethod :: !Text
  , rpcNotificationParams :: !(Maybe Value)
  , rpcNotificationJsonrpc :: !Text
  } deriving (Show, Generic)

-- | MCP Message wrapper
data MCPMessage
  = MCPRequest !JsonRpcRequest
  | MCPResponse !JsonRpcResponse
  | MCPNotification !JsonRpcNotification
  deriving (Show, Generic)

-- | MCP Capabilities
data MCPCapabilities = MCPCapabilities
  { capTools :: !Bool
  , capResources :: !Bool
  , capPrompts :: !Bool
  , capSampling :: !Bool
  , capRoots :: !Bool
  } deriving (Show, Generic)

-- | MCP Tool definition
data MCPTool = MCPTool
  { toolName :: !Text
  , toolDescription :: !Text
  , toolInputSchema :: !Value
  , toolOutputSchema :: !(Maybe Value)
  } deriving (Show, Generic)

-- | MCP Resource definition
data MCPResource = MCPResource
  { resourceUri :: !Text
  , resourceName :: !Text
  , resourceDescription :: !(Maybe Text)
  , resourceMimeType :: !(Maybe Text)
  } deriving (Show, Generic)

-- | MCP Prompt definition
data MCPPrompt = MCPPrompt
  { promptName :: !Text
  , promptDescription :: !Text
  , promptArguments :: ![Value]
  } deriving (Show, Generic)

-- | Transport type
data TransportType
  = StdioTransport
  | HttpTransport
  | WebSocketTransport
  deriving (Show, Eq, Generic)

-- | Transport configuration
data TransportConfig = TransportConfig
  { transportType :: !TransportType
  , transportHost :: !(Maybe Text)
  , transportPort :: !(Maybe Int)
  , transportPath :: !(Maybe Text)
  , transportSecure :: !Bool
  } deriving (Show, Generic)

-- | Connection state
data ConnectionState
  = Disconnected
  | Connecting
  | Connected
  | Authenticated
  | Error !Text
  deriving (Show, Eq, Generic)

-- | Server information
data ServerInfo = ServerInfo
  { serverInfoId :: !UUID
  , serverInfoName :: !Text
  , serverInfoVersion :: !Text
  , serverInfoCapabilities :: !ServerCapabilities
  , serverInfoTransport :: !TransportConfig
  , serverInfoLastSeen :: !UTCTime
  } deriving (Show, Generic)

-- | Server capabilities
data ServerCapabilities = ServerCapabilities
  { serverCapTools :: ![MCPTool]
  , serverCapResources :: ![MCPResource]
  , serverCapPrompts :: ![MCPPrompt]
  , serverCapSampling :: !Bool
  } deriving (Show, Generic)

-- | Discovery configuration
data DiscoveryConfig = DiscoveryConfig
  { discoveryEnabled :: !Bool
  , discoveryInterval :: !Int
  , discoveryTimeout :: !Int
  , discoveryRetries :: !Int
  } deriving (Show, Generic)

-- JSON instances
instance ToJSON JsonRpcRequest where
  toJSON req = object
    [ "jsonrpc" .= rpcJsonrpc req
    , "method" .= rpcMethod req
    , "params" .= rpcParams req
    , "id" .= rpcId req
    ]

instance FromJSON JsonRpcRequest where
  parseJSON = withObject "JsonRpcRequest" $ \o -> JsonRpcRequest
    <$> o .:? "id"
    <*> o .: "method"
    <*> o .:? "params"
    <*> o .: "jsonrpc"

instance ToJSON JsonRpcResponse where
  toJSON resp = object
    [ "jsonrpc" .= rpcResponseJsonrpc resp
    , "id" .= rpcResponseId resp
    , "result" .= rpcResult resp
    , "error" .= rpcError resp
    ]

instance FromJSON JsonRpcResponse where
  parseJSON = withObject "JsonRpcResponse" $ \o -> JsonRpcResponse
    <$> o .: "id"
    <*> o .:? "result"
    <*> o .:? "error"
    <*> o .: "jsonrpc"

instance ToJSON JsonRpcError
instance FromJSON JsonRpcError
instance ToJSON JsonRpcNotification
instance FromJSON JsonRpcNotification
instance ToJSON MCPMessage
instance FromJSON MCPMessage
instance ToJSON MCPCapabilities
instance FromJSON MCPCapabilities
instance ToJSON MCPTool
instance FromJSON MCPTool
instance ToJSON MCPResource
instance FromJSON MCPResource
instance ToJSON MCPPrompt
instance FromJSON MCPPrompt
instance ToJSON TransportType
instance FromJSON TransportType
instance ToJSON TransportConfig
instance FromJSON TransportConfig
instance ToJSON ConnectionState
instance FromJSON ConnectionState
instance ToJSON ServerInfo
instance FromJSON ServerInfo
instance ToJSON ServerCapabilities
instance FromJSON ServerCapabilities
instance ToJSON DiscoveryConfig
instance FromJSON DiscoveryConfig

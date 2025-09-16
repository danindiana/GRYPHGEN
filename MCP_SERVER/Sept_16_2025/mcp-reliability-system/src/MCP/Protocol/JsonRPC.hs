
{-# LANGUAGE OverloadedStrings #-}

module MCP.Protocol.JsonRPC
  ( -- * JSON-RPC Message Handling
    encodeRequest
  , encodeResponse
  , encodeNotification
  , decodeMessage
  , createRequest
  , createResponse
  , createErrorResponse
  , createNotification
  
    -- * Error Codes
  , parseError
  , invalidRequest
  , methodNotFound
  , invalidParams
  , internalError
  ) where

import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T
import MCP.Protocol.Types

-- | Standard JSON-RPC error codes
parseError, invalidRequest, methodNotFound, invalidParams, internalError :: Int
parseError = -32700
invalidRequest = -32600
methodNotFound = -32601
invalidParams = -32602
internalError = -32603

-- | Create a JSON-RPC request
createRequest :: Value -> Text -> Maybe Value -> JsonRpcRequest
createRequest reqId method params = JsonRpcRequest
  { rpcId = Just reqId
  , rpcMethod = method
  , rpcParams = params
  , rpcJsonrpc = "2.0"
  }

-- | Create a JSON-RPC response
createResponse :: Value -> Value -> JsonRpcResponse
createResponse respId result = JsonRpcResponse
  { rpcResponseId = respId
  , rpcResult = Just result
  , rpcError = Nothing
  , rpcResponseJsonrpc = "2.0"
  }

-- | Create a JSON-RPC error response
createErrorResponse :: Value -> Int -> Text -> Maybe Value -> JsonRpcResponse
createErrorResponse respId code message errorData = JsonRpcResponse
  { rpcResponseId = respId
  , rpcResult = Nothing
  , rpcError = Just $ JsonRpcError code message errorData
  , rpcResponseJsonrpc = "2.0"
  }

-- | Create a JSON-RPC notification
createNotification :: Text -> Maybe Value -> JsonRpcNotification
createNotification method params = JsonRpcNotification
  { rpcNotificationMethod = method
  , rpcNotificationParams = params
  , rpcNotificationJsonrpc = "2.0"
  }

-- | Encode a JSON-RPC request to JSON
encodeRequest :: JsonRpcRequest -> Value
encodeRequest = toJSON

-- | Encode a JSON-RPC response to JSON
encodeResponse :: JsonRpcResponse -> Value
encodeResponse = toJSON

-- | Encode a JSON-RPC notification to JSON
encodeNotification :: JsonRpcNotification -> Value
encodeNotification = toJSON

-- | Decode a JSON value to an MCP message
decodeMessage :: Value -> Either Text MCPMessage
decodeMessage value = do
  -- Try to parse as different message types
  case fromJSON value of
    Success req@(JsonRpcRequest {}) -> 
      if rpcId req == Nothing
        then Left "Request must have an id"
        else Right $ MCPRequest req
    Error _ -> case fromJSON value of
      Success resp@(JsonRpcResponse {}) -> Right $ MCPResponse resp
      Error _ -> case fromJSON value of
        Success notif@(JsonRpcNotification {}) -> Right $ MCPNotification notif
        Error err -> Left $ T.pack $ "Failed to parse JSON-RPC message: " ++ err

-- | Validate JSON-RPC message structure
validateMessage :: MCPMessage -> Either Text MCPMessage
validateMessage msg@(MCPRequest req) = do
  -- Validate request
  if T.null (rpcMethod req)
    then Left "Method cannot be empty"
    else if rpcJsonrpc req /= "2.0"
      then Left "Invalid JSON-RPC version"
      else Right msg

validateMessage msg@(MCPResponse resp) = do
  -- Validate response
  if rpcResponseJsonrpc resp /= "2.0"
    then Left "Invalid JSON-RPC version"
    else case (rpcResult resp, rpcError resp) of
      (Nothing, Nothing) -> Left "Response must have either result or error"
      (Just _, Just _) -> Left "Response cannot have both result and error"
      _ -> Right msg

validateMessage msg@(MCPNotification notif) = do
  -- Validate notification
  if T.null (rpcNotificationMethod notif)
    then Left "Method cannot be empty"
    else if rpcNotificationJsonrpc notif /= "2.0"
      then Left "Invalid JSON-RPC version"
      else Right msg

-- | Helper function to create common error responses
createParseErrorResponse :: Value -> JsonRpcResponse
createParseErrorResponse respId = createErrorResponse respId parseError "Parse error" Nothing

createInvalidRequestResponse :: Value -> JsonRpcResponse
createInvalidRequestResponse respId = createErrorResponse respId invalidRequest "Invalid Request" Nothing

createMethodNotFoundResponse :: Value -> Text -> JsonRpcResponse
createMethodNotFoundResponse respId method = 
  createErrorResponse respId methodNotFound "Method not found" (Just $ String method)

createInvalidParamsResponse :: Value -> Text -> JsonRpcResponse
createInvalidParamsResponse respId reason = 
  createErrorResponse respId invalidParams "Invalid params" (Just $ String reason)

createInternalErrorResponse :: Value -> Text -> JsonRpcResponse
createInternalErrorResponse respId reason = 
  createErrorResponse respId internalError "Internal error" (Just $ String reason)

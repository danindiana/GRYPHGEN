
{-# LANGUAGE OverloadedStrings #-}

module MCP.Benchmarks.Protocol
  ( jsonRpcBenchmarks
  , clientBenchmarks
  ) where

import Criterion.Main
import Control.DeepSeq
import Data.Aeson (Value(..), object, (.=))
import Data.Text (Text)
import qualified Data.Text as T

import MCP.Protocol.JsonRPC
import MCP.Protocol.Client
import MCP.Protocol.Types

-- | JSON-RPC benchmarks
jsonRpcBenchmarks :: [Benchmark]
jsonRpcBenchmarks =
  [ bench "create request" $ nf createTestRequest ()
  , bench "create response" $ nf createTestResponse ()
  , bench "encode request" $ nf encodeRequest testRequest
  , bench "decode message" $ nf decodeMessage testRequestValue
  ]

-- | Client benchmarks
clientBenchmarks :: [Benchmark]
clientBenchmarks =
  [ bench "create client" $ nfIO createTestClient
  , bench "send request" $ nfIO benchSendRequest
  ]

-- Test data
testRequest :: JsonRpcRequest
testRequest = createRequest (String "test-id") "test/method" (Just testParams)

testParams :: Value
testParams = object
  [ "param1" .= ("value1" :: Text)
  , "param2" .= (42 :: Int)
  ]

testRequestValue :: Value
testRequestValue = encodeRequest testRequest

testTransportConfig :: TransportConfig
testTransportConfig = TransportConfig
  { transportType = HttpTransport
  , transportHost = Just "localhost"
  , transportPort = Just 8080
  , transportPath = Just "/mcp"
  , transportSecure = False
  }

-- Benchmark implementations
createTestRequest :: () -> JsonRpcRequest
createTestRequest _ = testRequest

createTestResponse :: () -> JsonRpcResponse
createTestResponse _ = createResponse (String "test-id") (String "success")

createTestClient :: IO MCPClient
createTestClient = createClient testTransportConfig

benchSendRequest :: IO ()
benchSendRequest = do
  client <- createTestClient
  _ <- sendRequest client "test/method" (Just testParams)
  return ()

-- NFData instances
instance NFData JsonRpcRequest where
  rnf (JsonRpcRequest i m p j) = rnf i `seq` rnf m `seq` rnf p `seq` rnf j

instance NFData JsonRpcResponse where
  rnf (JsonRpcResponse i r e j) = rnf i `seq` rnf r `seq` rnf e `seq` rnf j

instance NFData JsonRpcError where
  rnf (JsonRpcError c m d) = rnf c `seq` rnf m `seq` rnf d

instance NFData TransportType where
  rnf StdioTransport = ()
  rnf HttpTransport = ()
  rnf WebSocketTransport = ()

instance NFData TransportConfig where
  rnf (TransportConfig t h p path s) = rnf t `seq` rnf h `seq` rnf p `seq` rnf path `seq` rnf s

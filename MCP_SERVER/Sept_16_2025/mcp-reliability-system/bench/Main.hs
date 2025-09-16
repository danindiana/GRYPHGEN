
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Criterion.Main
import Control.DeepSeq
import Data.Text (Text)
import qualified Data.Text as T
import System.Random

import MCP.Benchmarks.Reliability
import MCP.Benchmarks.Security
import MCP.Benchmarks.Protocol

main :: IO ()
main = defaultMain
  [ bgroup "Reliability"
      [ bgroup "CircuitBreaker" circuitBreakerBenchmarks
      , bgroup "Cache" cacheBenchmarks
      ]
  , bgroup "Security"
      [ bgroup "ParameterGuard" parameterGuardBenchmarks
      , bgroup "Sandbox" sandboxBenchmarks
      ]
  , bgroup "Protocol"
      [ bgroup "JsonRPC" jsonRpcBenchmarks
      , bgroup "Client" clientBenchmarks
      ]
  ]

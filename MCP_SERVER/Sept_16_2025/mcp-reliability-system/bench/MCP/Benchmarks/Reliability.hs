
{-# LANGUAGE OverloadedStrings #-}

module MCP.Benchmarks.Reliability
  ( circuitBreakerBenchmarks
  , cacheBenchmarks
  ) where

import Criterion.Main
import Control.DeepSeq
import Control.Exception (try, SomeException)
import Data.Text (Text)
import qualified Data.Text as T

import MCP.Reliability.CircuitBreaker
import MCP.Reliability.Cache
import MCP.Reliability.Types

-- | Circuit breaker benchmarks
circuitBreakerBenchmarks :: [Benchmark]
circuitBreakerBenchmarks =
  [ bench "create circuit breaker" $ nfIO createTestCircuitBreaker
  , bench "execute successful operation" $ nfIO benchSuccessfulExecution
  , bench "execute failing operation" $ nfIO benchFailingExecution
  , bench "check breaker state" $ nfIO benchStateCheck
  ]

-- | Cache benchmarks
cacheBenchmarks :: [Benchmark]
cacheBenchmarks =
  [ bench "create cache" $ nfIO createTestCache
  , bench "cache put operation" $ nfIO benchCachePut
  , bench "cache get hit" $ nfIO benchCacheGetHit
  , bench "cache get miss" $ nfIO benchCacheGetMiss
  , bench "cache cleanup" $ nfIO benchCacheCleanup
  ]

-- Circuit breaker benchmark implementations
createTestCircuitBreaker :: IO CircuitBreaker
createTestCircuitBreaker = do
  let config = CircuitBreakerConfig 5 30 60
  createCircuitBreaker config

benchSuccessfulExecution :: IO ()
benchSuccessfulExecution = do
  cb <- createTestCircuitBreaker
  _ <- executeWithBreaker cb (return "success")
  return ()

benchFailingExecution :: IO ()
benchFailingExecution = do
  cb <- createTestCircuitBreaker
  _ <- executeWithBreaker cb (error "test failure" :: IO String)
  return ()

benchStateCheck :: IO ()
benchStateCheck = do
  cb <- createTestCircuitBreaker
  _ <- getBreakerState cb
  return ()

-- Cache benchmark implementations
createTestCache :: IO (Cache Text Text)
createTestCache = do
  let config = CacheConfig 1000 300 60
  createCache config

benchCachePut :: IO ()
benchCachePut = do
  cache <- createTestCache
  cachePut cache "test-key" "test-value"

benchCacheGetHit :: IO ()
benchCacheGetHit = do
  cache <- createTestCache
  cachePut cache "test-key" "test-value"
  _ <- cacheGet cache "test-key"
  return ()

benchCacheGetMiss :: IO ()
benchCacheGetMiss = do
  cache <- createTestCache
  _ <- cacheGet cache "nonexistent-key"
  return ()

benchCacheCleanup :: IO ()
benchCacheCleanup = do
  cache <- createTestCache
  -- Add some entries
  mapM_ (\i -> cachePut cache (T.pack $ "key-" ++ show i) (T.pack $ "value-" ++ show i)) [1..100]
  cacheCleanup cache

-- NFData instances for benchmarking
instance NFData CircuitBreakerState
instance NFData CircuitBreakerConfig where
  rnf (CircuitBreakerConfig f t r) = f `seq` t `seq` r `seq` ()


{-# LANGUAGE DeriveGeneric #-}

module MCP.Reliability.Types
  ( -- * Circuit Breaker Types
    CircuitBreakerState(..)
  , CircuitBreakerConfig(..)
  , CircuitBreakerError(..)
  
    -- * Cache Types
  , CacheConfig(..)
  , CacheEntry(..)
  , CacheKey(..)
  
    -- * Fallback Types
  , FallbackConfig(..)
  , FallbackStrategy(..)
  
    -- * Metrics Types
  , MetricsConfig(..)
  , RequestMetrics(..)
  ) where

import Data.Aeson
import Data.Text (Text)
import Data.Time (UTCTime)
import GHC.Generics

-- | Circuit breaker state
data CircuitBreakerState
  = Closed
  | Open
  | HalfOpen
  deriving (Show, Eq, Generic)

-- | Circuit breaker configuration
data CircuitBreakerConfig = CircuitBreakerConfig
  { failureThreshold :: !Int
  , timeoutSeconds :: !Int
  , recoveryTimeout :: !Int
  } deriving (Show, Generic)

-- | Circuit breaker errors
data CircuitBreakerError
  = CircuitOpen !Text
  | TooManyFailures !Text
  | TimeoutExceeded !Text
  deriving (Show, Eq, Generic)

-- | Cache configuration
data CacheConfig = CacheConfig
  { maxSize :: !Int
  , ttlSeconds :: !Int
  , cleanupInterval :: !Int
  } deriving (Show, Generic)

-- | Cache entry
data CacheEntry a = CacheEntry
  { entryValue :: !a
  , entryTimestamp :: !UTCTime
  , entryTTL :: !Int
  } deriving (Show, Generic)

-- | Cache key
newtype CacheKey = CacheKey { unCacheKey :: Text }
  deriving (Show, Eq, Ord, Generic)

-- | Fallback configuration
data FallbackConfig = FallbackConfig
  { fallbackEnabled :: !Bool
  , maxFallbacks :: !Int
  , fallbackTimeout :: !Int
  } deriving (Show, Generic)

-- | Fallback strategy
data FallbackStrategy
  = RoundRobin
  | WeightedRandom
  | HealthBased
  deriving (Show, Eq, Generic)

-- | Metrics configuration
data MetricsConfig = MetricsConfig
  { metricsEnabled :: !Bool
  , metricsPort :: !Int
  , metricsPath :: !Text
  } deriving (Show, Generic)

-- | Request metrics
data RequestMetrics = RequestMetrics
  { requestCount :: !Int
  , successCount :: !Int
  , failureCount :: !Int
  , averageLatency :: !Double
  , p95Latency :: !Double
  , p99Latency :: !Double
  } deriving (Show, Generic)

-- JSON instances
instance ToJSON CircuitBreakerState
instance FromJSON CircuitBreakerState
instance ToJSON CircuitBreakerConfig
instance FromJSON CircuitBreakerConfig
instance ToJSON CircuitBreakerError
instance FromJSON CircuitBreakerError
instance ToJSON CacheConfig
instance FromJSON CacheConfig
instance (ToJSON a) => ToJSON (CacheEntry a)
instance (FromJSON a) => FromJSON (CacheEntry a)
instance ToJSON CacheKey
instance FromJSON CacheKey
instance ToJSON FallbackConfig
instance FromJSON FallbackConfig
instance ToJSON FallbackStrategy
instance FromJSON FallbackStrategy
instance ToJSON MetricsConfig
instance FromJSON MetricsConfig
instance ToJSON RequestMetrics
instance FromJSON RequestMetrics

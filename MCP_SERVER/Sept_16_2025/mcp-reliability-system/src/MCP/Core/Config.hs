
{-# LANGUAGE OverloadedStrings #-}

module MCP.Core.Config
  ( loadConfig
  , defaultConfig
  , validateConfig
  , ConfigError(..)
  ) where

import Control.Exception (Exception, throwIO)
import Data.Aeson
import Data.Text (Text)
import Data.Yaml (decodeFileEither)
import MCP.Core.Types

-- | Configuration loading errors
data ConfigError
  = ConfigFileNotFound FilePath
  | ConfigParseError Text
  | ConfigValidationError Text
  deriving (Show)

instance Exception ConfigError

-- | Load configuration from YAML file
loadConfig :: FilePath -> IO EngineConfig
loadConfig path = do
  result <- decodeFileEither path
  case result of
    Left err -> throwIO $ ConfigParseError (show err)
    Right config -> do
      validateConfig config
      return config

-- | Default configuration
defaultConfig :: EngineConfig
defaultConfig = EngineConfig
  { reliabilityConfig = ReliabilityConfig
      { circuitBreakerConfig = object
          [ "failure_threshold" .= (5 :: Int)
          , "timeout_seconds" .= (30 :: Int)
          , "recovery_timeout" .= (60 :: Int)
          ]
      , cacheConfig = object
          [ "max_size" .= (1000 :: Int)
          , "ttl_seconds" .= (300 :: Int)
          , "cleanup_interval" .= (60 :: Int)
          ]
      , fallbackConfig = object
          [ "enabled" .= True
          , "max_fallbacks" .= (3 :: Int)
          , "fallback_timeout" .= (10 :: Int)
          ]
      , retryConfig = object
          [ "max_retries" .= (3 :: Int)
          , "base_delay_ms" .= (100 :: Int)
          , "max_delay_ms" .= (5000 :: Int)
          ]
      }
  , securityConfig = SecurityConfig
      { sandboxEnabled = True
      , parameterGuardConfig = object
          [ "max_input_length" .= (10000 :: Int)
          , "allowed_patterns" .= (["^[a-zA-Z0-9_-]+$"] :: [Text])
          , "blocked_patterns" .= (["<script", "javascript:", "eval("] :: [Text])
          ]
      , permissionConfig = object
          [ "default_permissions" .= (["read"] :: [Text])
          , "admin_permissions" .= (["read", "write", "execute"] :: [Text])
          ]
      , auditLogging = True
      }
  , monitoringConfig = MonitoringConfig
      { prometheusEnabled = True
      , prometheusPort = 9090
      , loggingLevel = "INFO"
      , healthCheckEnabled = True
      }
  , maxConcurrentRequests = 100
  , requestTimeoutSeconds = 30
  }

-- | Validate configuration
validateConfig :: EngineConfig -> IO ()
validateConfig config = do
  -- Validate monitoring config
  let monitoring = monitoringConfig config
  when (prometheusPort monitoring < 1024 || prometheusPort monitoring > 65535) $
    throwIO $ ConfigValidationError "Prometheus port must be between 1024 and 65535"
  
  -- Validate request limits
  when (maxConcurrentRequests config <= 0) $
    throwIO $ ConfigValidationError "Max concurrent requests must be positive"
  
  when (requestTimeoutSeconds config <= 0) $
    throwIO $ ConfigValidationError "Request timeout must be positive"
  
  return ()

when :: Bool -> IO () -> IO ()
when True action = action
when False _ = return ()


{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module MCP.Core.Types
  ( -- * Core Types
    MCPEngine(..)
  , MCPRequest(..)
  , MCPResponse(..)
  , MCPError(..)
  , ToolId(..)
  , ServerId(..)
  , RequestId(..)
  
    -- * Configuration Types
  , EngineConfig(..)
  , ReliabilityConfig(..)
  , SecurityConfig(..)
  , MonitoringConfig(..)
  
    -- * Status Types
  , EngineStatus(..)
  , HealthStatus(..)
  ) where

import Data.Aeson
import Data.Text (Text)
import Data.Time (UTCTime)
import GHC.Generics
import Data.UUID (UUID)

-- | Core MCP Engine
data MCPEngine = MCPEngine
  { engineId :: !UUID
  , engineConfig :: !EngineConfig
  , engineStatus :: !EngineStatus
  , startTime :: !UTCTime
  } deriving (Show, Generic)

-- | MCP Request
data MCPRequest = MCPRequest
  { requestId :: !RequestId
  , toolId :: !ToolId
  , serverId :: !ServerId
  , parameters :: !Value
  , timestamp :: !UTCTime
  , metadata :: !(Maybe Value)
  } deriving (Show, Generic)

-- | MCP Response
data MCPResponse = MCPResponse
  { responseId :: !RequestId
  , result :: !(Either MCPError Value)
  , executionTime :: !Double
  , fromCache :: !Bool
  , serverId :: !ServerId
  } deriving (Show, Generic)

-- | MCP Error types
data MCPError
  = ValidationError !Text
  | SecurityError !Text
  | NetworkError !Text
  | TimeoutError !Text
  | CircuitBreakerOpen !Text
  | UnknownError !Text
  deriving (Show, Eq, Generic)

-- | Tool identifier
newtype ToolId = ToolId { unToolId :: Text }
  deriving (Show, Eq, Ord, Generic)

-- | Server identifier
newtype ServerId = ServerId { unServerId :: Text }
  deriving (Show, Eq, Ord, Generic)

-- | Request identifier
newtype RequestId = RequestId { unRequestId :: UUID }
  deriving (Show, Eq, Ord, Generic)

-- | Engine configuration
data EngineConfig = EngineConfig
  { reliabilityConfig :: !ReliabilityConfig
  , securityConfig :: !SecurityConfig
  , monitoringConfig :: !MonitoringConfig
  , maxConcurrentRequests :: !Int
  , requestTimeoutSeconds :: !Int
  } deriving (Show, Generic)

-- | Reliability configuration
data ReliabilityConfig = ReliabilityConfig
  { circuitBreakerConfig :: !Value
  , cacheConfig :: !Value
  , fallbackConfig :: !Value
  , retryConfig :: !Value
  } deriving (Show, Generic)

-- | Security configuration
data SecurityConfig = SecurityConfig
  { sandboxEnabled :: !Bool
  , parameterGuardConfig :: !Value
  , permissionConfig :: !Value
  , auditLogging :: !Bool
  } deriving (Show, Generic)

-- | Monitoring configuration
data MonitoringConfig = MonitoringConfig
  { prometheusEnabled :: !Bool
  , prometheusPort :: !Int
  , loggingLevel :: !Text
  , healthCheckEnabled :: !Bool
  } deriving (Show, Generic)

-- | Engine status
data EngineStatus
  = Starting
  | Running
  | Degraded
  | Stopping
  | Stopped
  deriving (Show, Eq, Generic)

-- | Health status
data HealthStatus = HealthStatus
  { healthy :: !Bool
  , checks :: ![Text]
  , lastCheck :: !UTCTime
  } deriving (Show, Generic)

-- JSON instances
instance ToJSON MCPEngine
instance FromJSON MCPEngine
instance ToJSON MCPRequest
instance FromJSON MCPRequest
instance ToJSON MCPResponse
instance FromJSON MCPResponse
instance ToJSON MCPError
instance FromJSON MCPError
instance ToJSON ToolId
instance FromJSON ToolId
instance ToJSON ServerId
instance FromJSON ServerId
instance ToJSON RequestId
instance FromJSON RequestId
instance ToJSON EngineConfig
instance FromJSON EngineConfig
instance ToJSON ReliabilityConfig
instance FromJSON ReliabilityConfig
instance ToJSON SecurityConfig
instance FromJSON SecurityConfig
instance ToJSON MonitoringConfig
instance FromJSON MonitoringConfig
instance ToJSON EngineStatus
instance FromJSON EngineStatus
instance ToJSON HealthStatus
instance FromJSON HealthStatus

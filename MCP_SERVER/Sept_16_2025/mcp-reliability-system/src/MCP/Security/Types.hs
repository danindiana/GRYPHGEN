
{-# LANGUAGE DeriveGeneric #-}

module MCP.Security.Types
  ( -- * Security Error Types
    SecurityError(..)
  , ValidationError(..)
  
    -- * Parameter Guard Types
  , ParameterGuardConfig(..)
  , InputPattern(..)
  
    -- * Sandbox Types
  , SandboxConfig(..)
  , SandboxResult(..)
  , SandboxError(..)
  
    -- * Permission Types
  , Permission(..)
  , PermissionSet(..)
  , PermissionConfig(..)
  , AccessLevel(..)
  ) where

import Data.Aeson
import Data.Set (Set)
import Data.Text (Text)
import GHC.Generics

-- | Security error types
data SecurityError
  = ParameterInjectionDetected !Text
  | UnauthorizedAccess !Text
  , SandboxViolation !Text
  | InputValidationFailed !Text
  | PermissionDenied !Text
  deriving (Show, Eq, Generic)

-- | Validation error details
data ValidationError = ValidationError
  { validationField :: !Text
  , validationMessage :: !Text
  , validationInput :: !Text
  } deriving (Show, Eq, Generic)

-- | Parameter guard configuration
data ParameterGuardConfig = ParameterGuardConfig
  { maxInputLength :: !Int
  , allowedPatterns :: ![InputPattern]
  , blockedPatterns :: ![InputPattern]
  , enableSqlInjectionCheck :: !Bool
  , enableXssCheck :: !Bool
  , enableCommandInjectionCheck :: !Bool
  } deriving (Show, Generic)

-- | Input validation pattern
data InputPattern = InputPattern
  { patternRegex :: !Text
  , patternDescription :: !Text
  , patternSeverity :: !Text
  } deriving (Show, Generic)

-- | Sandbox configuration
data SandboxConfig = SandboxConfig
  { sandboxEnabled :: !Bool
  , timeoutSeconds :: !Int
  , memoryLimitMB :: !Int
  , allowNetworkAccess :: !Bool
  , allowFileSystemAccess :: !Bool
  , allowedDirectories :: ![Text]
  , blockedCommands :: ![Text]
  } deriving (Show, Generic)

-- | Sandbox execution result
data SandboxResult a = SandboxResult
  { sandboxSuccess :: !Bool
  , sandboxResult :: !(Either SandboxError a)
  , sandboxExecutionTime :: !Double
  , sandboxMemoryUsed :: !Int
  } deriving (Show, Generic)

-- | Sandbox error types
data SandboxError
  = SandboxTimeout
  | MemoryLimitExceeded
  | UnauthorizedFileAccess !Text
  | UnauthorizedNetworkAccess !Text
  | BlockedCommandExecution !Text
  | SandboxSetupError !Text
  deriving (Show, Eq, Generic)

-- | Permission types
data Permission
  = ReadPermission
  | WritePermission
  | ExecutePermission
  | AdminPermission
  | CustomPermission !Text
  deriving (Show, Eq, Ord, Generic)

-- | Set of permissions
newtype PermissionSet = PermissionSet { unPermissionSet :: Set Permission }
  deriving (Show, Eq, Generic)

-- | Permission configuration
data PermissionConfig = PermissionConfig
  { defaultPermissions :: !PermissionSet
  , adminPermissions :: !PermissionSet
  , guestPermissions :: !PermissionSet
  , permissionInheritance :: !Bool
  } deriving (Show, Generic)

-- | Access level
data AccessLevel
  = NoAccess
  | ReadOnly
  | ReadWrite
  | FullAccess
  deriving (Show, Eq, Ord, Generic)

-- JSON instances
instance ToJSON SecurityError
instance FromJSON SecurityError
instance ToJSON ValidationError
instance FromJSON ValidationError
instance ToJSON ParameterGuardConfig
instance FromJSON ParameterGuardConfig
instance ToJSON InputPattern
instance FromJSON InputPattern
instance ToJSON SandboxConfig
instance FromJSON SandboxConfig
instance (ToJSON a) => ToJSON (SandboxResult a)
instance (FromJSON a) => FromJSON (SandboxResult a)
instance ToJSON SandboxError
instance FromJSON SandboxError
instance ToJSON Permission
instance FromJSON Permission
instance ToJSON PermissionSet
instance FromJSON PermissionSet
instance ToJSON PermissionConfig
instance FromJSON PermissionConfig
instance ToJSON AccessLevel
instance FromJSON AccessLevel

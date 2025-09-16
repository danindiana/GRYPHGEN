
{-# LANGUAGE OverloadedStrings #-}

module MCP.Benchmarks.Security
  ( parameterGuardBenchmarks
  , sandboxBenchmarks
  ) where

import Criterion.Main
import Control.DeepSeq
import Data.Text (Text)
import qualified Data.Text as T

import MCP.Security.ParameterGuard
import MCP.Security.Sandbox
import MCP.Security.Types

-- | Parameter guard benchmarks
parameterGuardBenchmarks :: [Benchmark]
parameterGuardBenchmarks =
  [ bench "create parameter guard" $ nfIO createTestParameterGuard
  , bench "validate safe input" $ nfIO benchValidateSafeInput
  , bench "validate malicious input" $ nfIO benchValidateMaliciousInput
  , bench "sanitize input" $ nf sanitizeInput testInput
  ]

-- | Sandbox benchmarks
sandboxBenchmarks :: [Benchmark]
sandboxBenchmarks =
  [ bench "create sandbox" $ nfIO createTestSandbox
  , bench "execute safe command" $ nfIO benchExecuteSafeCommand
  , bench "execute blocked command" $ nfIO benchExecuteBlockedCommand
  ]

-- Test data
testInput :: Text
testInput = "This is a test input with some special characters: <>&'\""

maliciousInput :: Text
maliciousInput = "<script>alert('xss')</script>"

-- Parameter guard benchmark implementations
createTestParameterGuard :: IO ParameterGuard
createTestParameterGuard = do
  let config = ParameterGuardConfig
        { maxInputLength = 10000
        , allowedPatterns = 
            [ InputPattern "^[a-zA-Z0-9_\\-\\.\\s]+$" "Alphanumeric" "low" ]
        , blockedPatterns = 
            [ InputPattern "<script" "Script tag" "high" ]
        , enableSqlInjectionCheck = True
        , enableXssCheck = True
        , enableCommandInjectionCheck = True
        }
  createParameterGuard config

benchValidateSafeInput :: IO ()
benchValidateSafeInput = do
  guard <- createTestParameterGuard
  _ <- validateInput guard "safe input 123"
  return ()

benchValidateMaliciousInput :: IO ()
benchValidateMaliciousInput = do
  guard <- createTestParameterGuard
  _ <- validateInput guard maliciousInput
  return ()

-- Sandbox benchmark implementations
createTestSandbox :: IO Sandbox
createTestSandbox = do
  let config = SandboxConfig
        { sandboxEnabled = True
        , timeoutSeconds = 30
        , memoryLimitMB = 512
        , allowNetworkAccess = False
        , allowFileSystemAccess = False
        , allowedDirectories = ["/tmp"]
        , blockedCommands = ["rm", "sudo"]
        }
  createSandbox config "test-sandbox"

benchExecuteSafeCommand :: IO ()
benchExecuteSafeCommand = do
  sandbox <- createTestSandbox
  _ <- executeSandboxed sandbox "echo" ["hello world"]
  return ()

benchExecuteBlockedCommand :: IO ()
benchExecuteBlockedCommand = do
  sandbox <- createTestSandbox
  _ <- executeSandboxed sandbox "rm" ["-rf", "/"]
  return ()

-- NFData instances
instance NFData SecurityError where
  rnf (ParameterInjectionDetected t) = rnf t
  rnf (UnauthorizedAccess t) = rnf t
  rnf (SandboxViolation t) = rnf t
  rnf (InputValidationFailed t) = rnf t
  rnf (PermissionDenied t) = rnf t

instance NFData InputPattern where
  rnf (InputPattern r d s) = rnf r `seq` rnf d `seq` rnf s

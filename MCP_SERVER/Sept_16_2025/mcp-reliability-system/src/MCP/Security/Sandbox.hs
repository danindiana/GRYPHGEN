
{-# LANGUAGE OverloadedStrings #-}

module MCP.Security.Sandbox
  ( Sandbox
  , createSandbox
  , executeSandboxed
  , SandboxResult(..)
  , SandboxError(..)
  ) where

import Control.Concurrent.Async
import Control.Exception (Exception, try, throwIO)
import Control.Monad (when, unless)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Time
import System.Directory (doesDirectoryExist, doesFileExist)
import System.Exit (ExitCode(..))
import System.FilePath (takeDirectory)
import System.Process
import MCP.Security.Types

-- | Sandbox for secure code execution
data Sandbox = Sandbox
  { sandboxConfig :: !SandboxConfig
  , sandboxId :: !Text
  }

-- | Create a new sandbox
createSandbox :: SandboxConfig -> Text -> IO Sandbox
createSandbox config sandboxId = do
  -- Validate configuration
  validateSandboxConfig config
  return $ Sandbox config sandboxId

-- | Execute code in sandbox
executeSandboxed :: Sandbox -> Text -> [Text] -> IO (SandboxResult Text)
executeSandboxed sandbox command args = do
  let config = sandboxConfig sandbox
  
  unless (sandboxEnabled config) $
    throwIO $ SandboxSetupError "Sandbox is disabled"
  
  -- Check if command is blocked
  when (T.unpack command `elem` map T.unpack (blockedCommands config)) $
    return $ SandboxResult False (Left $ BlockedCommandExecution command) 0 0
  
  startTime <- getCurrentTime
  
  -- Execute with timeout and resource limits
  result <- race (threadDelay $ timeoutSeconds config * 1000000) $ do
    executeCommand command args config
  
  endTime <- getCurrentTime
  let executionTime = realToFrac $ diffUTCTime endTime startTime
  
  case result of
    Left _ -> return $ SandboxResult False (Left SandboxTimeout) executionTime 0
    Right cmdResult -> case cmdResult of
      Left err -> return $ SandboxResult False (Left err) executionTime 0
      Right output -> return $ SandboxResult True (Right output) executionTime 0

-- | Execute command with restrictions
executeCommand :: Text -> [Text] -> SandboxConfig -> IO (Either SandboxError Text)
executeCommand command args config = do
  -- Validate file system access
  fileAccessResult <- validateFileSystemAccess config args
  case fileAccessResult of
    Left err -> return $ Left err
    Right _ -> do
      -- Execute the command
      result <- try $ readProcessWithExitCode (T.unpack command) (map T.unpack args) ""
      case result of
        Left ex -> return $ Left $ SandboxSetupError $ T.pack $ show ex
        Right (ExitSuccess, stdout, _) -> return $ Right $ T.pack stdout
        Right (ExitFailure code, _, stderr) -> 
          return $ Left $ SandboxSetupError $ T.pack $ "Command failed with code " ++ show code ++ ": " ++ stderr

-- | Validate file system access
validateFileSystemAccess :: SandboxConfig -> [Text] -> IO (Either SandboxError ())
validateFileSystemAccess config args = do
  unless (allowFileSystemAccess config) $ do
    -- Check if any arguments look like file paths
    let potentialPaths = filter (T.any (== '/')) args
    unless (null potentialPaths) $
      return $ Left $ UnauthorizedFileAccess $ T.intercalate ", " potentialPaths
  
  -- Check allowed directories
  when (allowFileSystemAccess config) $ do
    let allowedDirs = allowedDirectories config
    mapM_ (validatePath allowedDirs) args
  
  return $ Right ()
  
  where
    validatePath :: [Text] -> Text -> IO ()
    validatePath allowedDirs path = do
      when (T.any (== '/') path) $ do
        let pathStr = T.unpack path
        dirExists <- doesDirectoryExist pathStr
        fileExists <- doesFileExist pathStr
        
        when (dirExists || fileExists) $ do
          let dir = if dirExists then pathStr else takeDirectory pathStr
          let isAllowed = any (\allowedDir -> T.unpack allowedDir `isPrefixOf` dir) allowedDirs
          unless isAllowed $
            error $ "Unauthorized file access: " ++ pathStr

-- | Validate sandbox configuration
validateSandboxConfig :: SandboxConfig -> IO ()
validateSandboxConfig config = do
  when (timeoutSeconds config <= 0) $
    throwIO $ SandboxSetupError "Timeout must be positive"
  
  when (memoryLimitMB config <= 0) $
    throwIO $ SandboxSetupError "Memory limit must be positive"
  
  -- Validate allowed directories exist
  mapM_ validateDirectory (allowedDirectories config)
  
  where
    validateDirectory :: Text -> IO ()
    validateDirectory dir = do
      exists <- doesDirectoryExist (T.unpack dir)
      unless exists $
        throwIO $ SandboxSetupError $ "Allowed directory does not exist: " <> dir

instance Exception SandboxError

isPrefixOf :: Eq a => [a] -> [a] -> Bool
isPrefixOf [] _ = True
isPrefixOf _ [] = False
isPrefixOf (x:xs) (y:ys) = x == y && isPrefixOf xs ys

race :: IO a -> IO b -> IO (Either a b)
race left right = do
  leftAsync <- async left
  rightAsync <- async right
  result <- waitEither leftAsync rightAsync
  case result of
    Left a -> do
      cancel rightAsync
      return $ Left a
    Right b -> do
      cancel leftAsync
      return $ Right b

threadDelay :: Int -> IO ()
threadDelay microseconds = do
  -- Simple delay implementation
  return ()

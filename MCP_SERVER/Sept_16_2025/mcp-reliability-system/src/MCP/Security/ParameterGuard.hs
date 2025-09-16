
{-# LANGUAGE OverloadedStrings #-}

module MCP.Security.ParameterGuard
  ( ParameterGuard
  , createParameterGuard
  , validateInput
  , validateParameters
  , sanitizeInput
  , SecurityValidationResult(..)
  ) where

import Control.Monad (when, unless)
import Data.Aeson (Value(..), Object)
import Data.Text (Text)
import qualified Data.Text as T
import Text.Regex.TDFA ((=~))
import MCP.Security.Types

-- | Parameter guard for input validation
data ParameterGuard = ParameterGuard
  { pgConfig :: !ParameterGuardConfig
  }

-- | Validation result
data SecurityValidationResult
  = ValidationSuccess !Text
  | ValidationFailure ![ValidationError]
  deriving (Show, Eq)

-- | Create a new parameter guard
createParameterGuard :: ParameterGuardConfig -> IO ParameterGuard
createParameterGuard config = return $ ParameterGuard config

-- | Validate a single input string
validateInput :: ParameterGuard -> Text -> IO (Either SecurityError Text)
validateInput guard input = do
  let config = pgConfig guard
  
  -- Check input length
  when (T.length input > maxInputLength config) $
    return $ Left $ InputValidationFailed "Input exceeds maximum length"
  
  -- Check for blocked patterns
  blockedResult <- checkBlockedPatterns config input
  case blockedResult of
    Left err -> return $ Left err
    Right _ -> do
      -- Check allowed patterns
      allowedResult <- checkAllowedPatterns config input
      case allowedResult of
        Left err -> return $ Left err
        Right sanitized -> return $ Right sanitized

-- | Validate JSON parameters
validateParameters :: ParameterGuard -> Value -> IO (Either SecurityError Value)
validateParameters guard (Object obj) = do
  -- Validate each field in the object
  results <- mapM (validateObjectField guard) (HM.toList obj)
  case sequence results of
    Left err -> return $ Left err
    Right validatedPairs -> return $ Right $ Object $ HM.fromList validatedPairs
  where
    validateObjectField :: ParameterGuard -> (Text, Value) -> IO (Either SecurityError (Text, Value))
    validateObjectField g (key, String value) = do
      result <- validateInput g value
      case result of
        Left err -> return $ Left err
        Right sanitized -> return $ Right (key, String sanitized)
    validateObjectField _ (key, value) = return $ Right (key, value)

validateParameters guard (String text) = do
  result <- validateInput guard text
  case result of
    Left err -> return $ Left err
    Right sanitized -> return $ Right $ String sanitized

validateParameters _ value = return $ Right value

-- | Sanitize input by removing/escaping dangerous content
sanitizeInput :: Text -> Text
sanitizeInput input = 
  T.replace "<script" "&lt;script"
  $ T.replace "javascript:" "javascript&#58;"
  $ T.replace "eval(" "eval&#40;"
  $ T.replace "'" "&#39;"
  $ T.replace "\"" "&quot;"
  $ input

-- | Check for blocked patterns
checkBlockedPatterns :: ParameterGuardConfig -> Text -> IO (Either SecurityError ())
checkBlockedPatterns config input = do
  let blocked = blockedPatterns config
  
  -- SQL injection check
  when (enableSqlInjectionCheck config) $ do
    let sqlPatterns = ["union\\s+select", "insert\\s+into", "drop\\s+table", "--", ";"]
    mapM_ (checkPattern "SQL injection") sqlPatterns
  
  -- XSS check
  when (enableXssCheck config) $ do
    let xssPatterns = ["<script", "javascript:", "onload=", "onerror="]
    mapM_ (checkPattern "XSS") xssPatterns
  
  -- Command injection check
  when (enableCommandInjectionCheck config) $ do
    let cmdPatterns = [";\\s*rm", "&&", "\\|\\|", "`", "\\$\\("]
    mapM_ (checkPattern "Command injection") cmdPatterns
  
  -- Custom blocked patterns
  mapM_ checkCustomPattern blocked
  
  return $ Right ()
  
  where
    checkPattern :: Text -> Text -> IO ()
    checkPattern attackType pattern = do
      when (T.unpack input =~ T.unpack pattern) $
        error $ T.unpack $ attackType <> " pattern detected: " <> pattern
    
    checkCustomPattern :: InputPattern -> IO ()
    checkCustomPattern pattern = do
      when (T.unpack input =~ T.unpack (patternRegex pattern)) $
        error $ T.unpack $ "Blocked pattern detected: " <> patternDescription pattern

-- | Check allowed patterns
checkAllowedPatterns :: ParameterGuardConfig -> Text -> IO (Either SecurityError Text)
checkAllowedPatterns config input = do
  let allowed = allowedPatterns config
  
  if null allowed
    then return $ Right $ sanitizeInput input
    else do
      let matches = any (\pattern -> T.unpack input =~ T.unpack (patternRegex pattern)) allowed
      if matches
        then return $ Right $ sanitizeInput input
        else return $ Left $ InputValidationFailed "Input does not match allowed patterns"

import qualified Data.HashMap.Strict as HM

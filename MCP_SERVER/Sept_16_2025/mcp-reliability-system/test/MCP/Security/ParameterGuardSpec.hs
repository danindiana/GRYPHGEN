
{-# LANGUAGE OverloadedStrings #-}

module MCP.Security.ParameterGuardSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Text (Text)
import qualified Data.Text as T

import MCP.Security.ParameterGuard
import MCP.Security.Types

spec :: Spec
spec = describe "ParameterGuard" $ do
  
  let defaultConfig = ParameterGuardConfig
        { maxInputLength = 1000
        , allowedPatterns = 
            [ InputPattern "^[a-zA-Z0-9_\\-\\.\\s]+$" "Alphanumeric" "low" ]
        , blockedPatterns = 
            [ InputPattern "<script" "Script tag" "high"
            , InputPattern "javascript:" "JavaScript protocol" "high"
            ]
        , enableSqlInjectionCheck = True
        , enableXssCheck = True
        , enableCommandInjectionCheck = True
        }
  
  describe "validateInput" $ do
    it "accepts valid input" $ do
      guard <- createParameterGuard defaultConfig
      result <- validateInput guard "hello world 123"
      case result of
        Right _ -> return ()
        Left err -> expectationFailure $ "Expected success, got: " ++ show err
    
    it "rejects input exceeding max length" $ do
      let config = defaultConfig { maxInputLength = 5 }
      guard <- createParameterGuard config
      result <- validateInput guard "this is too long"
      case result of
        Left (InputValidationFailed _) -> return ()
        _ -> expectationFailure "Expected InputValidationFailed"
    
    it "detects script tag injection" $ do
      guard <- createParameterGuard defaultConfig
      result <- validateInput guard "<script>alert('xss')</script>"
      case result of
        Left _ -> return ()
        Right _ -> expectationFailure "Expected security error"
    
    it "detects JavaScript protocol" $ do
      guard <- createParameterGuard defaultConfig
      result <- validateInput guard "javascript:alert('xss')"
      case result of
        Left _ -> return ()
        Right _ -> expectationFailure "Expected security error"
    
    it "detects SQL injection patterns" $ do
      guard <- createParameterGuard defaultConfig
      result <- validateInput guard "'; DROP TABLE users; --"
      case result of
        Left _ -> return ()
        Right _ -> expectationFailure "Expected security error"
  
  describe "sanitizeInput" $ do
    it "escapes dangerous characters" $ do
      let input = "<script>alert('test')</script>"
      let sanitized = sanitizeInput input
      sanitized `shouldContain` "&lt;script"
    
    it "escapes quotes" $ do
      let input = "test'quote\"double"
      let sanitized = sanitizeInput input
      sanitized `shouldContain` "&#39;"
      sanitized `shouldContain` "&quot;"
  
  describe "property tests" $ do
    it "always sanitizes output" $ property $ \input -> do
      let sanitized = sanitizeInput (T.pack input)
      return $ not (T.isInfixOf "<script" sanitized)
    
    it "respects max length limit" $ property $ \maxLen input -> do
      maxLen > 0 && maxLen < 10000 ==> do
        let config = defaultConfig { maxInputLength = maxLen }
        guard <- createParameterGuard config
        result <- validateInput guard (T.pack input)
        case result of
          Left (InputValidationFailed _) -> return $ length input > maxLen
          Right _ -> return $ length input <= maxLen
          _ -> return True

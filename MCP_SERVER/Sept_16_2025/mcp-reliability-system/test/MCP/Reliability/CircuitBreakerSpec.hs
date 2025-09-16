
{-# LANGUAGE OverloadedStrings #-}

module MCP.Reliability.CircuitBreakerSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Control.Exception (throwIO)
import Control.Concurrent (threadDelay)

import MCP.Reliability.CircuitBreaker
import MCP.Reliability.Types

spec :: Spec
spec = describe "CircuitBreaker" $ do
  
  describe "createCircuitBreaker" $ do
    it "creates a circuit breaker in closed state" $ do
      let config = CircuitBreakerConfig 5 30 60
      cb <- createCircuitBreaker config
      state <- getBreakerState cb
      state `shouldBe` Closed
  
  describe "executeWithBreaker" $ do
    it "allows execution when circuit is closed" $ do
      let config = CircuitBreakerConfig 5 30 60
      cb <- createCircuitBreaker config
      result <- executeWithBreaker cb (return "success")
      result `shouldBe` Right "success"
    
    it "records failures and opens circuit" $ do
      let config = CircuitBreakerConfig 2 30 60
      cb <- createCircuitBreaker config
      
      -- First failure
      _ <- executeWithBreaker cb (throwIO $ userError "test error")
      state1 <- getBreakerState cb
      state1 `shouldBe` Closed
      
      -- Second failure should open circuit
      _ <- executeWithBreaker cb (throwIO $ userError "test error")
      state2 <- getBreakerState cb
      state2 `shouldBe` Open
    
    it "prevents execution when circuit is open" $ do
      let config = CircuitBreakerConfig 1 30 60
      cb <- createCircuitBreaker config
      
      -- Cause failure to open circuit
      _ <- executeWithBreaker cb (throwIO $ userError "test error")
      
      -- Next execution should be prevented
      result <- executeWithBreaker cb (return "success")
      case result of
        Left (CircuitOpen _) -> return ()
        _ -> expectationFailure "Expected CircuitOpen error"
  
  describe "resetBreaker" $ do
    it "resets circuit breaker to closed state" $ do
      let config = CircuitBreakerConfig 1 30 60
      cb <- createCircuitBreaker config
      
      -- Open the circuit
      _ <- executeWithBreaker cb (throwIO $ userError "test error")
      state1 <- getBreakerState cb
      state1 `shouldBe` Open
      
      -- Reset and verify
      resetBreaker cb
      state2 <- getBreakerState cb
      state2 `shouldBe` Closed

  describe "property tests" $ do
    it "maintains failure count correctly" $ property $ \failures -> do
      let config = CircuitBreakerConfig (failures + 1) 30 60
      cb <- createCircuitBreaker config
      
      -- Execute failures
      mapM_ (\_ -> executeWithBreaker cb (throwIO $ userError "test")) [1..failures]
      
      -- Circuit should still be closed
      state <- getBreakerState cb
      return $ state == Closed
    
    it "opens circuit after threshold failures" $ property $ \threshold -> do
      threshold > 0 ==> do
        let config = CircuitBreakerConfig threshold 30 60
        cb <- createCircuitBreaker config
        
        -- Execute threshold + 1 failures
        mapM_ (\_ -> executeWithBreaker cb (throwIO $ userError "test")) [1..(threshold + 1)]
        
        -- Circuit should be open
        state <- getBreakerState cb
        return $ state == Open


{-# LANGUAGE OverloadedStrings #-}

module MCP.Reliability.CircuitBreaker
  ( CircuitBreaker
  , createCircuitBreaker
  , executeWithBreaker
  , getBreakerState
  , resetBreaker
  , CircuitBreakerError(..)
  ) where

import Control.Concurrent.STM
import Control.Exception (Exception, throwIO, try)
import Control.Monad (when)
import Data.Time
import MCP.Reliability.Types

-- | Circuit breaker implementation
data CircuitBreaker = CircuitBreaker
  { cbConfig :: !CircuitBreakerConfig
  , cbState :: !(TVar CircuitBreakerState)
  , cbFailureCount :: !(TVar Int)
  , cbLastFailureTime :: !(TVar (Maybe UTCTime))
  , cbSuccessCount :: !(TVar Int)
  }

-- | Create a new circuit breaker
createCircuitBreaker :: CircuitBreakerConfig -> IO CircuitBreaker
createCircuitBreaker config = do
  state <- newTVarIO Closed
  failures <- newTVarIO 0
  lastFailure <- newTVarIO Nothing
  successes <- newTVarIO 0
  return $ CircuitBreaker config state failures lastFailure successes

-- | Execute an action with circuit breaker protection
executeWithBreaker :: CircuitBreaker -> IO a -> IO (Either CircuitBreakerError a)
executeWithBreaker cb action = do
  canExecute <- checkCanExecute cb
  if canExecute
    then do
      result <- try action
      case result of
        Left ex -> do
          recordFailure cb
          return $ Left $ TooManyFailures $ show ex
        Right value -> do
          recordSuccess cb
          return $ Right value
    else return $ Left $ CircuitOpen "Circuit breaker is open"

-- | Check if the circuit breaker allows execution
checkCanExecute :: CircuitBreaker -> IO Bool
checkCanExecute cb = atomically $ do
  state <- readTVar (cbState cb)
  case state of
    Closed -> return True
    Open -> do
      -- Check if we should transition to half-open
      now <- unsafeIOToSTM getCurrentTime
      lastFailure <- readTVar (cbLastFailureTime cb)
      case lastFailure of
        Nothing -> return False
        Just failTime -> do
          let recoveryTime = addUTCTime (fromIntegral $ recoveryTimeout $ cbConfig cb) failTime
          if now >= recoveryTime
            then do
              writeTVar (cbState cb) HalfOpen
              writeTVar (cbSuccessCount cb) 0
              return True
            else return False
    HalfOpen -> do
      successCount <- readTVar (cbSuccessCount cb)
      return $ successCount < 3 -- Allow limited requests in half-open state

-- | Record a failure
recordFailure :: CircuitBreaker -> IO ()
recordFailure cb = do
  now <- getCurrentTime
  atomically $ do
    failures <- readTVar (cbFailureCount cb)
    let newFailures = failures + 1
    writeTVar (cbFailureCount cb) newFailures
    writeTVar (cbLastFailureTime cb) (Just now)
    
    when (newFailures >= failureThreshold (cbConfig cb)) $
      writeTVar (cbState cb) Open

-- | Record a success
recordSuccess :: CircuitBreaker -> IO ()
recordSuccess cb = atomically $ do
  state <- readTVar (cbState cb)
  case state of
    Closed -> do
      writeTVar (cbFailureCount cb) 0
    HalfOpen -> do
      successes <- readTVar (cbSuccessCount cb)
      let newSuccesses = successes + 1
      writeTVar (cbSuccessCount cb) newSuccesses
      when (newSuccesses >= 3) $ do
        writeTVar (cbState cb) Closed
        writeTVar (cbFailureCount cb) 0
    Open -> return () -- Should not happen

-- | Get current circuit breaker state
getBreakerState :: CircuitBreaker -> IO CircuitBreakerState
getBreakerState cb = readTVarIO (cbState cb)

-- | Reset circuit breaker to closed state
resetBreaker :: CircuitBreaker -> IO ()
resetBreaker cb = atomically $ do
  writeTVar (cbState cb) Closed
  writeTVar (cbFailureCount cb) 0
  writeTVar (cbLastFailureTime cb) Nothing
  writeTVar (cbSuccessCount cb) 0

instance Exception CircuitBreakerError

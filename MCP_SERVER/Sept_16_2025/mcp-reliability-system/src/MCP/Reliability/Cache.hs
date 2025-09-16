
{-# LANGUAGE OverloadedStrings #-}

module MCP.Reliability.Cache
  ( Cache
  , createCache
  , cacheGet
  , cachePut
  , cacheInvalidate
  , cacheCleanup
  , cacheStats
  ) where

import Control.Concurrent.STM
import Control.Monad (when, void)
import Data.Hashable (Hashable)
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import Data.Time
import MCP.Reliability.Types

-- | Cache implementation with TTL support
data Cache k v = Cache
  { cacheConfig :: !CacheConfig
  , cacheData :: !(TVar (HashMap k (CacheEntry v)))
  , cacheStats :: !(TVar CacheStats)
  }

-- | Cache statistics
data CacheStats = CacheStats
  { cacheHits :: !Int
  , cacheMisses :: !Int
  , cacheEvictions :: !Int
  , cacheSize :: !Int
  } deriving (Show)

-- | Create a new cache
createCache :: (Hashable k, Eq k) => CacheConfig -> IO (Cache k v)
createCache config = do
  dataVar <- newTVarIO HM.empty
  statsVar <- newTVarIO $ CacheStats 0 0 0 0
  return $ Cache config dataVar statsVar

-- | Get value from cache
cacheGet :: (Hashable k, Eq k) => Cache k v -> k -> IO (Maybe v)
cacheGet cache key = do
  now <- getCurrentTime
  atomically $ do
    cacheMap <- readTVar (cacheData cache)
    stats <- readTVar (cacheStats cache)
    
    case HM.lookup key cacheMap of
      Nothing -> do
        writeTVar (cacheStats cache) stats { cacheMisses = cacheMisses stats + 1 }
        return Nothing
      Just entry -> do
        let expiryTime = addUTCTime (fromIntegral $ entryTTL entry) (entryTimestamp entry)
        if now > expiryTime
          then do
            -- Entry expired, remove it
            let newMap = HM.delete key cacheMap
            writeTVar (cacheData cache) newMap
            writeTVar (cacheStats cache) stats 
              { cacheMisses = cacheMisses stats + 1
              , cacheEvictions = cacheEvictions stats + 1
              , cacheSize = cacheSize stats - 1
              }
            return Nothing
          else do
            -- Entry valid, return it
            writeTVar (cacheStats cache) stats { cacheHits = cacheHits stats + 1 }
            return $ Just $ entryValue entry

-- | Put value in cache
cachePut :: (Hashable k, Eq k) => Cache k v -> k -> v -> IO ()
cachePut cache key value = do
  now <- getCurrentTime
  atomically $ do
    cacheMap <- readTVar (cacheData cache)
    stats <- readTVar (cacheStats cache)
    
    let entry = CacheEntry value now (ttlSeconds $ cacheConfig cache)
        newMap = HM.insert key entry cacheMap
        currentSize = HM.size cacheMap
        maxSize = maxSize $ cacheConfig cache
    
    -- Check if we need to evict entries
    finalMap <- if currentSize >= maxSize
      then do
        -- Simple LRU eviction - remove oldest entries
        let sortedEntries = HM.toList newMap
            evictCount = currentSize - maxSize + 1
            (toEvict, toKeep) = splitAt evictCount sortedEntries
            finalMap = HM.fromList toKeep
        writeTVar (cacheStats cache) stats 
          { cacheEvictions = cacheEvictions stats + evictCount
          , cacheSize = HM.size finalMap
          }
        return finalMap
      else do
        writeTVar (cacheStats cache) stats { cacheSize = HM.size newMap }
        return newMap
    
    writeTVar (cacheData cache) finalMap

-- | Invalidate cache entry
cacheInvalidate :: (Hashable k, Eq k) => Cache k v -> k -> IO ()
cacheInvalidate cache key = atomically $ do
  cacheMap <- readTVar (cacheData cache)
  stats <- readTVar (cacheStats cache)
  
  when (HM.member key cacheMap) $ do
    let newMap = HM.delete key cacheMap
    writeTVar (cacheData cache) newMap
    writeTVar (cacheStats cache) stats 
      { cacheEvictions = cacheEvictions stats + 1
      , cacheSize = cacheSize stats - 1
      }

-- | Clean up expired entries
cacheCleanup :: (Hashable k, Eq k) => Cache k v -> IO ()
cacheCleanup cache = do
  now <- getCurrentTime
  atomically $ do
    cacheMap <- readTVar (cacheData cache)
    stats <- readTVar (cacheStats cache)
    
    let isExpired entry = 
          let expiryTime = addUTCTime (fromIntegral $ entryTTL entry) (entryTimestamp entry)
          in now > expiryTime
        
        (expired, valid) = HM.partition isExpired cacheMap
        evictedCount = HM.size expired
    
    when (evictedCount > 0) $ do
      writeTVar (cacheData cache) valid
      writeTVar (cacheStats cache) stats 
        { cacheEvictions = cacheEvictions stats + evictedCount
        , cacheSize = HM.size valid
        }

-- | Get cache statistics
cacheStats :: Cache k v -> IO CacheStats
cacheStats cache = readTVarIO (cacheStats cache)

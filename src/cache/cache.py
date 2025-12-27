import diskcache
import logging
from logging import config
import pickle
import re
import sys
import time
#import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")
summary = logging.getLogger("summary")


class Cache():
    """
        prefixで名前空間を分けることができる．
        prefixはCache("filename", prefix="...")で指定できるほか，
        set(key, val, prefix="...")で他のインスタンスへのアクセスもできる．
        filenameがNoneか""なら，キャッシュはオフとなる
        オフかどうかは，is_enableで判断してもらう
        というか，アクセスしてきてもキャッシュが無いって返事すれば良いのか
        いや，それだけだと，無かったら計算してキャッシュに入れるとか無駄な処理が走るか？
        キャッシュに入れるもsetが呼ばれるだけだからら，内部でチェックしてスルーでもいいかもn
    """

    key_prefix = None
    is_enable = True
    def __init__(self,
                 cfg: DictConfig,
                 cache_filename,
                 prefix = None,
                ):
        """ __init__
        Args:
            cfg: Config object. (required)
            cache_filename: A name of the file on the local storage
                            where the item is stored. (required)
 
        """
        if cfg is not None:
            # cfgがDictConfigの場合、.get()でデフォルト値を指定して安全に取得
            top_dir = cfg.get("top_dir", "CacheStorage")
            root = cfg.get("root", "")
            enable = cfg.get("enable", True)
            self.overwrite = cfg.get("overwrite", False)

        if prefix is not None:
            self.key_prefix = prefix
        else:
            self.key_prefix = root
        
        logger.debug(f"Creating a cache object. key_prefix={self.key_prefix}, saves to {cache_filename}, enable={enable}, overwrite={self.overwrite}")
        
        if enable:
            if self.key_prefix:
                filename = f"{top_dir}/{self.key_prefix}/{cache_filename}"
            else:
                filename = f"{top_dir}/{cache_filename}"
            filename = filename.replace("//", "/")
            
            logger.debug(f"Cache file = {filename}")
            self.is_enable = True
            self.db = diskcache.Cache(filename)
        else:
            self.is_enable = False
            self.db = None
        logger.debug(f"db={self.db}")

    def close(self):
        self.db.close()

    # prefixがclassの識別子で，modeがclassの中のmethodやdataの識別子？
    def set(self,
            key: str,
            val: str | object):
        # if val.__class__.__name__ == "str":
        #     ret = self.db.set(key, val)
        # else:
        #     serialized_val = pickle.dumps(val)
        #     ret = self.db.set(key, serialized_val)
        # #self.db.sync()
        # 流石に自動的にシリアライズされるっぽいので修正
        ret = self.db.set(key, val)
        return ret

    def get(self, key):
        # val = self.db.get(key)
        # logger.debug(f"val={val}({val.__class__}")
        # if val.__class__.__name__ == "bytes":
        #     val = pickle.loads(val)
        # 流石に自動的にデシリアライズされるっぽいので修正
        val = self.db.get(key)
        #logger.debug(f"key={key}")
        #logger.debug(f"val={val}")
        return val

    def listKeys(self, regexp=".*"):
        key = regexp.replace("[", "\\[")
        key = key.replace("]", "\\]")
        """
        Return list of keys as an array.
        Args:
            regexp: (Optional) The key that matches regexp is returned.
        """
        iter = self.db.iterkeys(reverse=False)
        substring = f"{key}"

        hit_keys = []
        #logger.debug(f"target key={substring}")
        count = 0
        for key in iter:
            logger.debug(f"target record={key} class={type(key)}")
            logger.debug(f"target regexp={substring}")
            #print(re.search(substring, key))
            if isinstance(key, str):
                if re.search(substring, key):
                    logger.debug("match!")
                    hit_keys.append(key)
            else:
                logger.warn(f"Skip: key = {key}, class = {key.__class__}. Only supported class is string.")
                
            count += 1
        logger.debug(f"total number of items = {count}")
        return hit_keys

    def listKeysVals(self, regexp = ".*"):
        """
        Return list of keys as an array.
        Args:
            regexp: (Optional) The key that matches regexp is returned.
        """
        keyList = self.listKeys(regexp)
        keyValList = [(key, self.get(key)) for key in keyList]
        return keyValList
        
    def delete(self, regexp = None):
        """
        return: Counter indicating the number of deleted items
        """
        
        keys = self.listKeys(regexp)

        count = 0
        logger.info(f"delete target keys[0:5]={keys[0:5]}")
        #input("OK?")
        for key in keys:
            logger.debug(f"deleting candidate key={key}")
            #input("OK?")
            if self.db.delete(key):
                count += 1
        logger.info(f"deleted {count} records")
        return count

    def deleteAll(self):
        return self.delete(".*")

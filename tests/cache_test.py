from cache.cache import Cache
from hydra import initialize, compose
import polars as pl
import pytest

class TestCache:
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="exp_dev.yaml")
        cache = Cache(cfg=cfg,
                      cache_filename="cache_test",
                      prefix="test",
                      )
        another_cache = Cache(cfg=cfg,
                      cache_filename="cache_test",
                      prefix="test1",
                      )
        
    def test_create_object(self):
        assert self.cache is not None

    def test_set(self):
        ret = self.cache.set("test_key1", "test_val1")
        assert ret is True

    def test_get(self):
        ret = self.cache.set("test_key1", "test_val1")
        assert ret is True
        val = self.cache.get("test_key1")
        assert val == "test_val1"
        val = self.cache.get("test_key9")
        assert val is None

    def test_get_prefix(self):
        ret = self.cache.set("test_key1", "test_val1")
        assert ret is True
        ret = self.another_cache.set("test_key1", "test_val100")
        assert ret is True
        val = self.cache.get("test_key1")
        assert val == "test_val1"
        val = self.cache.get("test_key9")
        assert val is None
        val = self.another_cache.get("test_key1")
        assert val == "test_val100"
        val = self.another_cache.get("test_key9")
        assert val is None
        
    def test_listKeys(self):
        self.cache.deleteAll()
        ret = self.cache.set("test_key1", "test_val1")
        assert ret is True
        ret = self.cache.set("test_key2", "test_val2")
        assert ret is True
        ret = self.cache.set("test_key22", "test_val22")
        assert ret is True
        keys = self.cache.listKeys()
        assert keys == ["test_key1", "test_key2", "test_key22"]
        keys = self.cache.listKeys(regexp="test_key2.*")
        assert keys == ["test_key2", "test_key22"]
        keys = self.cache.listKeys(regexp="test_key2")
        assert keys == ["test_key2", "test_key22"]
    def test_listKeysVals(self):
        self.cache.deleteAll()
        ret = self.cache.set("test_key1", "test_val1")
        assert ret is True
        ret = self.cache.set("test_key2", "test_val2")
        assert ret is True
        ret = self.cache.set("test_key22", "test_val22")
        assert ret is True
        keysvals = self.cache.listKeysVals()
        assert keysvals == [("test_key1", "test_val1"),
                        ("test_key2", "test_val2"),
                        ("test_key22", "test_val22")]
        keysvals = self.cache.listKeysVals(regexp="test_key2.*")
        assert keysvals == [
                        ("test_key2", "test_val2"),
                        ("test_key22", "test_val22")]

        keysvals = self.cache.listKeysVals(regexp="test_key2")
        assert keysvals == [
                        ("test_key2", "test_val2"),
                        ("test_key22", "test_val22")]

        
    def test_delete(self):
        def data_set():
            ret = self.cache.set("test_key1", "test_val1")
            assert ret is True
            val = self.cache.get("test_key1")
            assert val == "test_val1"
            ret = self.cache.set("test_key2", "test_val2")
            assert ret is True
            val = self.cache.get("test_key2")
            assert val == "test_val2"
            ret = self.cache.set("test_key22", "test_val22")
            assert ret is True
            val = self.cache.get("test_key22")
            assert val == "test_val22"

            ret = self.another_cache.set("test_key1", "test_val100")
            assert ret is True
            val = self.another_cache.get("test_key1")
            assert val == "test_val100"
            ret = self.another_cache.set("test_key2", "test_val200")
            assert ret is True
            val = self.another_cache.get("test_key2")
            assert val == "test_val200"
            ret = self.another_cache.set("test_key22", "test_val2200")
            assert ret is True
            val = self.another_cache.get("test_key22")
            assert val == "test_val2200"

        # All clear
        data_set()
        self.cache.delete(".*")
        list = self.cache.listKeys()
        assert len(list) == 0
        list = self.another_cache.listKeys()
        assert len(list) == 3

        # One clear
        data_set()
        self.cache.delete("test_key2$")
        list = self.cache.listKeys()
        assert list == ["test_key1", "test_key22"]

        # Two clear
        data_set()
        self.cache.delete("test_key2")
        list = self.cache.listKeys()
        assert list == ["test_key1"]

    def test_binarydata(self):


        human = Human("Bob", 23)
        assert human.hello() == "Hello, my name is Bob"

        # b'\x80\x04\x955\x00\x00\x00\x00\x00\x00\x00\x8c\ncache_test\x94\x8c\x05Human\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94\x8c\x03Bob\x94\x8c\x03age\x94K\x17ub.'
        serialized_human = pickle.dumps(human)

        ret = self.cache.set("Human1", human)
        assert ret is True
        ret = self.cache.get("Human1")
        assert ret is not None

        #human = pickle.loads(ret)
        assert ret.name == "Bob"
        assert ret.hello() == "Hello, my name is Bob"

        
from dataclasses import dataclass
import pickle
@dataclass
class Human():
    name: str
    age: int
    def hello(self) -> str:
        return f"Hello, my name is {self.name}"


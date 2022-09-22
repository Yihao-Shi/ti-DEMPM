import taichi as ti


@ti.data_oriented
class HASHMAP:
    def __init__(self, HashSize, TensorSize, type):
        self.HashSize = HashSize
        if type == 0:
            self.map = ti.field(float, HashSize)
        elif type == 1:
            self.map = ti.Vector.field(TensorSize, float, HashSize)
        elif type == 2:
            self.map = ti.Matrix.field(TensorSize, TensorSize, float, HashSize)
        self.Key = ti.field(ti.u64, HashSize)
    
    @ti.func
    def rehash(self, key, attempt):
        return key + (attempt * ((key + 1) % self.HashSize + 1)) % self.HashSize

    @ti.kernel
    def MapInit(self):
        for key in self.Key:
            self.Key[key] = -1

    @ti.func
    def hash(self, key):
        return key % self.HashSize

    @ti.func
    def isConflict(self):
        pass

    @ti.func
    def Insert(self, key, value):
        slot = self.hash(key)
        if self.Key[slot] >= 0:
            attempt = 1
            while self.Key[slot] != -1:
                slot = self.rehash(key, attempt)
                attempt += 1
        self.Key[slot] = key
        self.map[slot] = value

    @ti.func
    def Search(self, key):
        slot = self.hash(key)
        attempt = 1
        while self.Key[slot] != key:
            slot = self.rehash(key, attempt)
        return slot

    @ti.func
    def Locate(self, slot):
        return self.map[slot]


import numpy as np


class RedisQueue:
    def __init__(self, redisDB):
        self.redisDB = redisDB

    def add(self, imageIdx, hist):
        p = self.redisDB.pipeline()

        for i in np.where(hist > 0)[0]:
            p.rpush("vw:{}".format(i), imageIdx)

        p.execute()

    def finish(self):
        self.redisDB.save()

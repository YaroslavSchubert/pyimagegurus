from __future__ import print_function
from db.redisqueue import RedisQueue
from redis import Redis
import argparse
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bovw-db", required=True, help='Path to bag-of-vis-words db')
args = vars(ap.parse_args())

redisDB = Redis(host='localhost', port=6379, db=11)
rq = RedisQueue(redisDB)

bovwDB = h5py.File(args['bovw_db'], mode='r')

for (i, hist) in enumerate(bovwDB['bovw']):
    if i > 0 and i % 10 == 0:
        print("[PROGRESS] processed {} entries".format(i))

    rq.add(i, hist)

bovwDB.close()
rq.finish()



# python build_redis_index.py --bovw-db output/bovw.hdf5
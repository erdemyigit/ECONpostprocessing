from dask.distributed import Client
from dask_jobqueue import SLURMCluster

def setup_cluster_on_submit(minjobs, maxjobs):
    import random

    uuid = random.getrandbits(64)
    print("saving logs with uuid %016x"%uuid)

    cluster = SLURMCluster(queue = 'submit,submit-gpu,submit-gpu1080',
                           cores=1,
                           processes=1,
                           memory='32GB',
                           walltime='01:00:00',
                           log_directory='logs/%016x'%uuid)
    cluster.adapt(minimum_jobs = minjobs, maximum_jobs = maxjobs)
    #cluster.scale(maxjobs)
    client = Client(cluster)
    return cluster, client

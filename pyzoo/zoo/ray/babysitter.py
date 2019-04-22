from zoo.ray import start_ray_driver

# for local mode
spark_home = "/home/lizhichao/bin/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/lizhichao/bin/jdk1.8.0_101/"
python_loc = "/home/lizhichao/anaconda3/envs/py36/bin/python"
python_zip_file = "/home/lizhichao/bin/god/analytics-zoo/pyzoo.zip"

# local node-018
# from ray_poc.test.allreduce.test_resnet_yarn import test_resnet_ps
#
# spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
# java_home = "/home/zhichao/god/jdk1.8.0_101/"
# python_loc = "/home/zhichao/anaconda2/envs/ray36/bin/python"
# extra_pmodule_zip = "/home/zhichao/god/analytics-zoo/pyzoo.zip"


from zoo.ray import RayRunner
spark_runner = SparkRunner(spark_home=spark_home, java_home=java_home)

num_workers = 4

sc  = spark_runner.run_on_local(python_loc=python_loc,
                                           python_zip_file=python_zip_file,
                                           driver_memory="10g",
                                           driver_cores=num_workers + 1) # plus 1 for master

rayRunner = RayRunner(sc=sc)

redis_address = rayRunner.run()

start_ray_driver(redis_address=redis_address)

# test_sgd(batch_size=128, num_workers=num_workers, grad_shard_bytes=85000000)
# test_resnet_ps(num_worker=num_workers)
print(redis_address)


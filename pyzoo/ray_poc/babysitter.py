from ray_poc.RayRunner import RayRunner
from ray_poc.util.spark import init_spark_on_yarn

sc, python_loc = init_spark_on_yarn()

rayRunner = RayRunner(sc, python_loc)

redis_address = rayRunner.run()

print(redis_address)

# TODO: setting the python log? or just print or redirect to elsewhere?

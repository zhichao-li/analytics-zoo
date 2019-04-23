import ray
import time

ray.init()

@ray.remote
class Test():
    def test_sleep(self):
        time.sleep(20)
        print("hello")
        return("hello")

    def test_sleep2(self):
        print("hello2")
        return "hello2"

@ray.remote
class Parameters():
    def get(self, *t):
        return "parameters"

ta = Test.remote()
pa = Parameters.remote()

for i in range(0, 10):
    r1 = ta.test_sleep.remote()
    r2 = ta.test_sleep2.remote()
    pa.get.remote(r1, r2)
    print("here")
    # print(ray.get(r))



#include "memkind.h"

#include <memory>
#include <cstring>
#include <stdio.h>
#include <random>
#include <iostream>
#include <malloc.h>
#include <vector>
#include <stdlib.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <iostream>


#define BUFFER_SIZE 1024
#define PMEM_MAX_SIZE ((size_t)1024 * 1024 * 1024 * 128)

int main(int argc, char * argv[])
{
      std::random_device rd;
      std::mt19937 mt(rd());
      std::uniform_int_distribution<> m_size(0, 9);
      //std::uniform_int_distribution<> m_index(0, 10);

      double block_size[] = {8519680, 4325376, 8519680, 4325376, 8519680, 4325376, 8519680, 4325376, 432517, 608478};

        struct memkind *pmem_kind;
        int err = 0;

        err = memkind_create_pmem("/mnt/pmem0", PMEM_MAX_SIZE, &pmem_kind);
        if (err)
        {
            std::stringstream ss;
            char msg[200];
            memkind_error_message(err, msg, 200);
            ss << "memkind_create_pmem failed! error code is: " << err << msg <<"\n";
            std::cout<<ss<<"\n";

                printf("%s\n", "Unable to create pmem partition\n");
                return -1;
        } else {
           std::cout<<"sucess" << err <<"\n";
        }

        char *pmem_str = NULL;

       bool full = false;
      int evict = 0;
        long long malloc_sum = 0;

      std::vector<char*> pmem_strs;
      long long total = 128000000000 / 1024 / 1024;

        while (true)
        {
            int index = m_size(mt);
            size_t size = block_size[index];
                pmem_str = (char *)memkind_malloc(pmem_kind, size);
                if (pmem_str == NULL)
                {
                  //full = true;
                        printf("Unable to allocate pmem string (pmem_str)\n");
                  break;
                }
                else
                {
                  int length = pmem_strs.size() / 2;
                  std::uniform_int_distribution<> m_index(0, length-1);
                  long long usable_size = memkind_malloc_usable_size(pmem_kind, pmem_str);
                  while (usable_size / 1024 / 1024 + malloc_sum > total) {
                        int to_evict = m_index(mt);
                        char *str_to_evict = pmem_strs[to_evict];
                        long long evict_size = memkind_malloc_usable_size(pmem_kind, str_to_evict);
                        //printf("Evict size %d\n", evict_size);
                        malloc_sum -= evict_size / 1024 / 1024;
                        memkind_free(pmem_kind, str_to_evict);
                        pmem_strs.erase(pmem_strs.begin() + to_evict);
                        printf("Eviction occured %d\n", malloc_sum);
                  }
                  pmem_strs.push_back(pmem_str);
                  //printf("usable size %d\n", usable_size);
                  malloc_sum = usable_size / 1024 / 1024 + malloc_sum;
                        //printf("malloc sum %d\n", malloc_sum);
                }
            //system("df -k | grep pmem0");
        }
        //printf("%d\t\t%d\n", PMEM_MAX_SIZE / 1024 / 1024 / 1024, malloc_sum / 1024 / 1024 / 1024);

      printf("Out of memory");
        return 0;
}

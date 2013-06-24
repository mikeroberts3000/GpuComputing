import math
import numpy
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler

class ReduceManager:
    
    source_module = pycuda.compiler.SourceModule \
    (
    """
    __global__ void reduce_sum( float* d_scratchpad, int n, int num_threads )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
 
        int left_index  = global_index_1d;
        int right_index = global_index_1d + num_threads;

        if ( right_index < n )
        {
            d_scratchpad[ left_index ] = d_scratchpad[ left_index ] + d_scratchpad[ right_index ];
        }
    }

    __global__ void reduce_product( float* d_scratchpad, int n, int num_threads )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
 
        int left_index  = global_index_1d;
        int right_index = global_index_1d + num_threads;

        if ( right_index < n )
        {
            d_scratchpad[ left_index ] = d_scratchpad[ left_index ] * d_scratchpad[ right_index ];
        }
    }

    __global__ void reduce_min( float* d_scratchpad, int n, int num_threads )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
 
        int left_index  = global_index_1d;
        int right_index = global_index_1d + num_threads;

        if ( right_index < n )
        {
            d_scratchpad[ left_index ] = min( d_scratchpad[ left_index ], d_scratchpad[ right_index ] );
        }
    }

    __global__ void reduce_max( float* d_scratchpad, int n, int num_threads )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
 
        int left_index  = global_index_1d;
        int right_index = global_index_1d + num_threads;

        if ( right_index < n )
        {
            d_scratchpad[ left_index ] = max( d_scratchpad[ left_index ], d_scratchpad[ right_index ] );
        }
    }    
    """
    )

    _reduce_sum_function     = source_module.get_function("reduce_sum")
    _reduce_product_function = source_module.get_function("reduce_product")
    _reduce_min_function     = source_module.get_function("reduce_min")
    _reduce_max_function     = source_module.get_function("reduce_max")

    _size_of_element_bytes           = 4
    _block_size_num_elements         = 1024
    _block_size_num_threads          = _block_size_num_elements / 2

    _max_num_elements                = -1
    _n                               = -1
    _scratchpad_device               = -1

    def __init__(self, max_num_elements):
        
        self._max_num_elements          = max_num_elements
        self._num_bytes                 = self._max_num_elements * self._size_of_element_bytes
        self._scratchpad_device         = pycuda.driver.mem_alloc(self._num_bytes)

    def __copy_input_htod(self, input_data_host):

        assert input_data_host.shape[0] <= self._max_num_elements
        assert input_data_host.dtype    == numpy.float32

        pycuda.driver.memcpy_htod(self._scratchpad_device, input_data_host)

    def __copy_input_dtod(self, input_data_device, num_elements):

        pycuda.driver.memcpy_dtod(self._scratchpad_device, input_data_device, int(num_elements * self._size_of_element_bytes))
        
    def __reduce(self, num_elements, reduce_function):

        self._n = num_elements

        num_sweep_passes    = int(math.ceil(math.log(num_elements,2)))
        reduce_num_elements = self._n
        
        for d in range(num_sweep_passes):

            reduce_num_threads    = int(math.ceil(float(reduce_num_elements) / float(2)))
            
            reduce_function_block = (self._block_size_num_threads,1,1)
            num_blocks            = int(math.ceil(float(reduce_num_threads) / float(reduce_function_block[0])))
            reduce_function_grid  = (num_blocks, 1)
            
            reduce_function(
                self._scratchpad_device,
                numpy.int32(reduce_num_elements),
                numpy.int32(reduce_num_threads),                
                block=reduce_function_block,
                grid=reduce_function_grid)

            reduce_num_elements = reduce_num_threads

        tmp = numpy.zeros(1, dtype=numpy.float32)

        pycuda.driver.memcpy_dtoh(tmp, self._scratchpad_device)

        return tmp[0]

    def reduce_sum_device(self, input_data_device, num_elements):

        self.__copy_input_dtod(input_data_device, num_elements)
        return self.__reduce(num_elements, self._reduce_sum_function)

    def reduce_product_device(self, input_data_device, num_elements):

        self.__copy_input_dtod(input_data_device, num_elements)
        return self.__reduce(num_elements, self._reduce_product_function)

    def reduce_min_device(self, input_data_device, num_elements):

        self.__copy_input_dtod(input_data_device, num_elements)
        return self.__reduce(num_elements, self._reduce_min_function)

    def reduce_max_device(self, input_data_device, num_elements):

        self.__copy_input_dtod(input_data_device, num_elements)
        return self.__reduce(num_elements, self._reduce_max_function)

    def reduce_sum_host(self, input_data_host):

        num_elements = input_data_host.shape[0]
        
        self.__copy_input_htod(input_data_host)
        return self.__reduce(num_elements, self._reduce_sum_function)

    def reduce_product_host(self, input_data_host):

        num_elements = input_data_host.shape[0]
        
        self.__copy_input_htod(input_data_host)
        return self.__reduce(num_elements, self._reduce_product_function)

    def reduce_min_host(self, input_data_host):

        num_elements = input_data_host.shape[0]
        
        self.__copy_input_htod(input_data_host)
        return self.__reduce(num_elements, self._reduce_min_function)

    def reduce_max_host(self, input_data_host):

        num_elements = input_data_host.shape[0]
        
        self.__copy_input_htod(input_data_host)
        return self.__reduce(num_elements, self._reduce_max_function)

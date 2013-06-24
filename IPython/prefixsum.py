import math
import numpy
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler

class PrefixSumManager:
    
    source_module = pycuda.compiler.SourceModule \
    (
    """
    __global__ void prefix_sum_up_sweep( unsigned int* d_prefix_sum, int n, int d )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int k               = global_index_1d * ( 2 << d );

        int left_index;
        int right_index;

        if ( d == 0 )
        {
            left_index  = k;
            right_index = k + 1;
        }
        else
        {
            left_index  = k + ( 2 << ( d - 1 ) ) - 1;
            right_index = k + ( 2 << d )         - 1;
        }

        if ( right_index < n )
        {
            d_prefix_sum[ right_index ] = d_prefix_sum[ left_index ] + d_prefix_sum[ right_index ];
        }
    }

    __global__ void prefix_sum_down_sweep( unsigned int* d_prefix_sum, int n, int d )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int k               = global_index_1d * ( 2 << d );

        int left_index;
        int right_index;

        if ( d == 0 )
        {
            left_index  = k;
            right_index = k + 1;
        }
        else
        {
            left_index  = k + ( 2 << ( d - 1 ) ) - 1;
            right_index = k + ( 2 << d )         - 1;
        }

        if ( right_index < n )
        {
            unsigned int temp           = d_prefix_sum[ right_index ];
            d_prefix_sum[ right_index ] = d_prefix_sum[ left_index ] + d_prefix_sum[ right_index ];
            d_prefix_sum[ left_index ]  = temp;
        }
    }

    __global__ void blocked_prefix_sum_set_last_block_elements_to_zero( unsigned int* d_prefix_sums, int n, int block_size_num_elements )
    {
        int global_index_1d_left  = ( ( ( threadIdx.x * 2 ) + 1 ) * block_size_num_elements ) - 1;
        int global_index_1d_right = ( ( ( threadIdx.x * 2 ) + 2 ) * block_size_num_elements ) - 1;

        if ( global_index_1d_left < n )
        {
            d_prefix_sums[ global_index_1d_left ] = 0;
        }

        if ( global_index_1d_right < n )
        {
            d_prefix_sums[ global_index_1d_right ] = 0;
        }
    }

    __global__ void blocked_prefix_sum_down_sweep( unsigned int* d_prefix_sum, unsigned int* d_block_sums, unsigned int* d_input_data_resized, int n, int d )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int k               = global_index_1d * ( 2 << d );

        int left_index;
        int right_index;

        if ( d == 0 )
        {
            left_index  = k;
            right_index = k + 1;
        }
        else
        {
            left_index  = k + ( 2 << ( d - 1 ) ) - 1;
            right_index = k + ( 2 << d )         - 1;
        }

        if ( right_index < n )
        {
            unsigned int temp           = d_prefix_sum[ right_index ];
            d_prefix_sum[ right_index ] = d_prefix_sum[ left_index ] + d_prefix_sum[ right_index ];
            d_prefix_sum[ left_index ]  = temp;
        }

        if ( d == 0 && threadIdx.x == blockDim.x - 1 )
        {
            d_block_sums[ blockIdx.x ] = d_prefix_sum[ right_index ] + d_input_data_resized[ right_index ];
        }
    }

    __global__ void blocked_prefix_sum_add_block_sums( unsigned int* d_prefix_sums, unsigned int* d_block_sums, int n )
    {
        int global_index_1d = 2 * ( ( blockIdx.x * blockDim.x ) + threadIdx.x );

        if ( blockIdx.x > 0 && global_index_1d < n - 1 )
        {
            unsigned int block_sum               = d_block_sums[ blockIdx.x ];
            d_prefix_sums[ global_index_1d ]     = d_prefix_sums[ global_index_1d ] + block_sum;
            d_prefix_sums[ global_index_1d + 1 ] = d_prefix_sums[ global_index_1d + 1 ] + block_sum;
        }
    }
    """
    )

    _prefix_sum_down_sweep_function                              = source_module.get_function("prefix_sum_down_sweep")
    _prefix_sum_up_sweep_function                                = source_module.get_function("prefix_sum_up_sweep")
    _blocked_prefix_sum_down_sweep_function                      = source_module.get_function("blocked_prefix_sum_down_sweep")
    _blocked_prefix_sum_set_last_block_elements_to_zero_function = source_module.get_function("blocked_prefix_sum_set_last_block_elements_to_zero")
    _blocked_prefix_sum_add_block_sums_function                  = source_module.get_function("blocked_prefix_sum_add_block_sums")

    _size_of_element_bytes           = 4
    _block_size_num_elements         = 1024
    _block_size_num_threads          = _block_size_num_elements / 2
    _num_sweep_passes                = int(math.ceil(math.log(_block_size_num_elements,2)))

    _max_num_elements                = -1
    _n                               = -1
    _input_data_resized_num_elements = -1
    _input_data_resized_num_threads  = -1
    _input_data_device               = -1
    _input_data_resized_device       = -1
    _prefix_sum_device               = -1
    _block_sums_device               = -1

    def __init__(self, max_num_elements):

        num_elements_to_pad = 0
        if max_num_elements % self._block_size_num_elements != 0:
            num_elements_to_pad = self._block_size_num_elements - (max_num_elements % self._block_size_num_elements)

        max_num_elements = max_num_elements + num_elements_to_pad
        
        assert max_num_elements <= self._block_size_num_elements**2
        
        self._max_num_elements          = max_num_elements
        self._num_bytes                 = self._max_num_elements * self._size_of_element_bytes
        self._input_data_device         = pycuda.driver.mem_alloc(self._num_bytes)
        self._input_data_resized_device = pycuda.driver.mem_alloc(self._num_bytes)
        self._output_data_device        = pycuda.driver.mem_alloc(self._num_bytes)
        self._prefix_sum_device         = pycuda.driver.mem_alloc(self._num_bytes)
        self._block_sums_device         = pycuda.driver.mem_alloc(self._num_bytes)

    def __copy_input_htod(self, input_data_host):

        assert input_data_host.shape[0] <  self._block_size_num_elements**2
        assert input_data_host.shape[0] <= self._max_num_elements
        assert input_data_host.dtype    == numpy.uint32

        pycuda.driver.memcpy_htod(self._input_data_device, input_data_host)
        
    def __initialize_prefix_sum(self, input_data_device, num_elements):

        self._n = num_elements

        num_elements_to_pad = 0
        if self._n % self._block_size_num_elements != 0:
            num_elements_to_pad = self._block_size_num_elements - (self._n % self._block_size_num_elements)

        self._input_data_resized_num_elements = self._n + num_elements_to_pad
        self._input_data_resized_num_threads  = self._input_data_resized_num_elements / 2

        assert self._input_data_resized_num_elements <= self._max_num_elements
        
        pycuda.driver.memset_d32(self._input_data_resized_device, 0, self._input_data_resized_num_elements)
        pycuda.driver.memset_d32(self._prefix_sum_device,         0, self._input_data_resized_num_elements)
        pycuda.driver.memset_d32(self._block_sums_device,         0, self._block_size_num_elements)

        pycuda.driver.memcpy_dtod(self._input_data_resized_device, input_data_device, self._n * self._size_of_element_bytes)
        pycuda.driver.memcpy_dtod(self._prefix_sum_device,         input_data_device, self._n * self._size_of_element_bytes)
        
    def __block_prefix_sum_input(self):
        
        prefix_sum_up_sweep_function_block = (self._block_size_num_threads,1,1)
        num_blocks                         = int(math.ceil(float(self._input_data_resized_num_threads) / float(prefix_sum_up_sweep_function_block[0])))
        prefix_sum_up_sweep_function_grid  = (num_blocks, 1)

        blocked_prefix_sum_set_last_block_elements_to_zero_function_block = (self._block_size_num_threads,1,1)
        num_blocks = int(math.ceil(float(self._block_size_num_threads) / float(blocked_prefix_sum_set_last_block_elements_to_zero_function_block[0])))
        blocked_prefix_sum_set_last_block_elements_to_zero_function_grid  = (num_blocks, 1)

        blocked_prefix_sum_down_sweep_function_block = (self._block_size_num_threads,1,1)
        num_blocks                                   = int(math.ceil(float(self._input_data_resized_num_threads) / float(blocked_prefix_sum_down_sweep_function_block[0])))
        blocked_prefix_sum_down_sweep_function_grid  = (num_blocks, 1)

        for d in range(self._num_sweep_passes):
            self._prefix_sum_up_sweep_function(
                self._prefix_sum_device,
                numpy.int32(self._input_data_resized_num_elements),
                numpy.int32(d),
                block=prefix_sum_up_sweep_function_block,
                grid=prefix_sum_up_sweep_function_grid)

        self._blocked_prefix_sum_set_last_block_elements_to_zero_function(
            self._prefix_sum_device,
            numpy.int32(self._input_data_resized_num_elements),
            numpy.int32(self._block_size_num_elements),
            block=blocked_prefix_sum_set_last_block_elements_to_zero_function_block,
            grid=blocked_prefix_sum_set_last_block_elements_to_zero_function_grid)

        for d in range(self._num_sweep_passes - 1,-1,-1):
            self._blocked_prefix_sum_down_sweep_function(
                self._prefix_sum_device,
                self._block_sums_device,
                self._input_data_resized_device,
                numpy.int32(self._input_data_resized_num_elements),
                numpy.int32(d),
                block=blocked_prefix_sum_down_sweep_function_block,
                grid=blocked_prefix_sum_down_sweep_function_grid)
    
    def __block_prefix_sum_block_sums(self):
        
        prefix_sum_up_sweep_function_block = (self._block_size_num_threads,1,1)
        num_blocks                         = int(math.ceil(float(self._block_size_num_threads) / float(prefix_sum_up_sweep_function_block[0])))
        prefix_sum_up_sweep_function_grid  = (num_blocks, 1)

        blocked_prefix_sum_set_last_block_elements_to_zero_function_block = (self._block_size_num_threads,1,1)
        num_blocks = int(math.ceil(float(self._block_size_num_threads) / float(blocked_prefix_sum_set_last_block_elements_to_zero_function_block[0])))
        blocked_prefix_sum_set_last_block_elements_to_zero_function_grid  = (num_blocks, 1)

        prefix_sum_down_sweep_function_block = (self._block_size_num_threads,1,1)
        num_blocks                           = int(math.ceil(float(self._block_size_num_threads) / float(prefix_sum_down_sweep_function_block[0])))
        prefix_sum_down_sweep_function_grid  = (num_blocks, 1)

        for d in range(self._num_sweep_passes):
            self._prefix_sum_up_sweep_function(
                self._block_sums_device,
                numpy.int32(self._block_size_num_elements),
                numpy.int32(d),
                block=prefix_sum_up_sweep_function_block,
                grid=prefix_sum_up_sweep_function_grid)

        self._blocked_prefix_sum_set_last_block_elements_to_zero_function(
            self._block_sums_device,
            numpy.int32(self._block_size_num_elements),
            numpy.int32(self._block_size_num_elements),
            block=blocked_prefix_sum_set_last_block_elements_to_zero_function_block,
            grid=blocked_prefix_sum_set_last_block_elements_to_zero_function_grid)

        for d in range(self._num_sweep_passes - 1,-1,-1):
            self._prefix_sum_down_sweep_function(
                self._block_sums_device,
                numpy.int32(self._block_size_num_elements),
                numpy.int32(d),
                block=prefix_sum_down_sweep_function_block,
                grid=prefix_sum_down_sweep_function_grid)

    def __distribute_block_sums(self):
        
        blocked_prefix_sum_add_block_sums_function_block = (self._block_size_num_threads,1,1)
        num_blocks                                       = int(math.ceil(float(self._input_data_resized_num_threads) / float(blocked_prefix_sum_add_block_sums_function_block[0])))
        blocked_prefix_sum_add_block_sums_function_grid  = (num_blocks, 1)

        self._blocked_prefix_sum_add_block_sums_function(
            self._prefix_sum_device,
            self._block_sums_device,
            numpy.int32(self._input_data_resized_num_elements),
            block=blocked_prefix_sum_add_block_sums_function_block,
            grid=blocked_prefix_sum_add_block_sums_function_grid)

    def __copy_output_dtod(self, output_data_device):

        pycuda.driver.memcpy_dtod(output_data_device, self._prefix_sum_device, self._n * self._size_of_element_bytes)
        
    def __copy_output_dtoh(self, output_data_host):

        pycuda.driver.memcpy_dtoh(output_data_host, self._prefix_sum_device)

    def prefix_sum_device(self, input_data_device, output_data_device, num_elements):

        self.__initialize_prefix_sum(input_data_device, num_elements)
        self.__block_prefix_sum_input()
        self.__block_prefix_sum_block_sums()
        self.__distribute_block_sums()
        self.__copy_output_dtod(output_data_device)

    def prefix_sum_host(self, input_data_host, output_data_host):

        num_elements = input_data_host.shape[0]
        
        self.__copy_input_htod(input_data_host)
        self.prefix_sum_device(self._input_data_device, self._output_data_device, num_elements)
        self.__copy_output_dtoh(output_data_host)

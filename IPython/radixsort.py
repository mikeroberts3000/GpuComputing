import math
import numpy
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler
import split

class RadixSortManager:
    
    source_module = pycuda.compiler.SourceModule \
    (
    """
    __global__ void radix_sort_compute_flags_ascending(
        unsigned int* d_input_data,
        unsigned int* d_output_data,
        int mask,
        int n )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

        if ( global_index_1d < n )
        {
            unsigned int input_value = d_input_data[ global_index_1d ];

            if ( input_value & mask )
            {
                d_output_data[ global_index_1d ] = 0;
            }
            else
            {
                d_output_data[ global_index_1d ] = 1;
            }
        }
    }

    __global__ void radix_sort_compute_flags_descending(
        unsigned int* d_input_data,
        unsigned int* d_output_data,
        int mask,
        int n )
    {
        int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

        if ( global_index_1d < n )
        {
            unsigned int input_value = d_input_data[ global_index_1d ];

            if ( input_value & mask )
            {
                d_output_data[ global_index_1d ] = 1;
            }
            else
            {
                d_output_data[ global_index_1d ] = 0;
            }
        }
    }    
    """
    )

    _radix_sort_compute_flags_ascending_function  = source_module.get_function("radix_sort_compute_flags_ascending")
    _radix_sort_compute_flags_descending_function = source_module.get_function("radix_sort_compute_flags_descending")

    _size_of_element_bytes   = 4
    _size_of_element_bits    = 32
    
    _max_num_elements        = -1
    _num_bytes               = -1

    _input_keys_device       = -1
    _input_values_device     = -1
    _flag_data_device        = -1
    _split_keys_old_device   = -1
    _split_values_old_device = -1
    _split_keys_new_device   = -1
    _split_values_new_device = -1

    _split_manager           = -1
    
    def __init__(self, max_num_elements):
        
        self._max_num_elements        = max_num_elements
        self._num_bytes               = self._max_num_elements * self._size_of_element_bytes

        self._input_keys_device       = pycuda.driver.mem_alloc(self._num_bytes)
        self._input_values_device     = pycuda.driver.mem_alloc(self._num_bytes)
        self._flag_data_device        = pycuda.driver.mem_alloc(self._num_bytes)
        self._split_keys_old_device   = pycuda.driver.mem_alloc(self._num_bytes)
        self._split_values_old_device = pycuda.driver.mem_alloc(self._num_bytes)
        self._output_keys_device      = pycuda.driver.mem_alloc(self._num_bytes)
        self._output_values_device    = pycuda.driver.mem_alloc(self._num_bytes)

        self._split_manager           = split.SplitManager(max_num_elements)

    def __copy_input_htod_key(self, input_keys_host):

        assert input_keys_host.shape[0] <= self._max_num_elements

        assert \
            input_keys_host.dtype == numpy.uint32  or \
            input_keys_host.dtype == numpy.int32   or \
            input_keys_host.dtype == numpy.float32

        pycuda.driver.memcpy_htod(self._input_keys_device,   input_keys_host)

    def __copy_input_htod_key_value(self, input_keys_host, input_values_host):

        assert input_keys_host.shape[0] == input_values_host.shape[0]
        assert input_keys_host.shape[0] <= self._max_num_elements

        assert \
            input_keys_host.dtype == numpy.uint32  or \
            input_keys_host.dtype == numpy.int32   or \
            input_keys_host.dtype == numpy.float32

        assert \
            input_values_host.dtype == numpy.uint32  or \
            input_values_host.dtype == numpy.int32   or \
            input_values_host.dtype == numpy.float32

        pycuda.driver.memcpy_htod(self._input_keys_device,   input_keys_host)
        pycuda.driver.memcpy_htod(self._input_values_device, input_values_host)

    def __radix_sort_key(self, input_keys_device, output_keys_device, num_elements, compute_flags_function):

        assert num_elements <= self._max_num_elements

        self._n = num_elements

        pycuda.driver.memcpy_dtod(self._split_keys_old_device, input_keys_device, self._n * self._size_of_element_bytes)
        
        pycuda.driver.memset_d32(self._flag_data_device, 0, self._n)
        pycuda.driver.memset_d32(output_keys_device,     0, self._n)

        for b in range(self._size_of_element_bits):

            mask = numpy.int32(2**numpy.int8(b))

            radix_sort_compute_flags_funcion_block = (512,1,1)
            num_blocks                             = int(math.ceil(float(self._n) / float(radix_sort_compute_flags_funcion_block[0])))
            radix_sort_compute_flags_funcion_grid  = (num_blocks, 1)
            
            compute_flags_function(
                self._split_keys_old_device,
                self._flag_data_device,
                numpy.int32(mask),
                numpy.int32(self._n),
                block=radix_sort_compute_flags_funcion_block,
                grid=radix_sort_compute_flags_funcion_grid)

            self._split_manager.split_device(self._split_keys_old_device, self._flag_data_device, output_keys_device, self._n)

            self._split_keys_old_device, output_keys_device = output_keys_device, self._split_keys_old_device

        pycuda.driver.memcpy_dtod(output_keys_device, self._split_keys_old_device, self._n * self._size_of_element_bytes)
        
    def __radix_sort_key_value(self, input_keys_device, input_values_device, output_keys_device, output_values_device, num_elements, compute_flags_function):

        assert num_elements <= self._max_num_elements

        self._n = num_elements

        pycuda.driver.memcpy_dtod(self._split_keys_old_device,   input_keys_device,   self._n * self._size_of_element_bytes)
        pycuda.driver.memcpy_dtod(self._split_values_old_device, input_values_device, self._n * self._size_of_element_bytes)
        
        pycuda.driver.memset_d32(self._flag_data_device, 0, self._n)
        pycuda.driver.memset_d32(output_keys_device,     0, self._n)
        pycuda.driver.memset_d32(output_values_device,   0, self._n)

        for b in range(self._size_of_element_bits):

            mask = numpy.int32(2**numpy.int8(b))

            radix_sort_compute_flags_funcion_block = (512,1,1)
            num_blocks                             = int(math.ceil(float(self._n) / float(radix_sort_compute_flags_funcion_block[0])))
            radix_sort_compute_flags_funcion_grid  = (num_blocks, 1)
            
            compute_flags_function(
                self._split_keys_old_device,
                self._flag_data_device,
                numpy.int32(mask),
                numpy.int32(self._n),
                block=radix_sort_compute_flags_funcion_block,
                grid=radix_sort_compute_flags_funcion_grid)

            self._split_manager.split_device(self._split_keys_old_device,   self._flag_data_device, output_keys_device,   self._n)
            self._split_manager.split_device(self._split_values_old_device, self._flag_data_device, output_values_device, self._n)

            self._split_keys_old_device,   output_keys_device   = output_keys_device,   self._split_keys_old_device
            self._split_values_old_device, output_values_device = output_values_device, self._split_values_old_device

        pycuda.driver.memcpy_dtod(output_keys_device,   self._split_keys_old_device,   self._n * self._size_of_element_bytes)
        pycuda.driver.memcpy_dtod(output_values_device, self._split_values_old_device, self._n * self._size_of_element_bytes)

    def __copy_output_dtoh_key(self, output_keys_host):

        pycuda.driver.memcpy_dtoh(output_keys_host, self._output_keys_device)

    def __copy_output_dtoh_key_value(self, output_keys_host, output_values_host):

        pycuda.driver.memcpy_dtoh(output_keys_host,   self._output_keys_device)
        pycuda.driver.memcpy_dtoh(output_values_host, self._output_values_device)

    def radix_sort_key_ascending_device(self, input_keys_device, output_keys_device, num_elements):

        self.__radix_sort_key(
            input_keys_device,
            output_keys_device,
            num_elements,
            self._radix_sort_compute_flags_ascending_function)

    def radix_sort_key_descending_device(self, input_keys_device, output_keys_device, num_elements):

        self.__radix_sort_key(
            input_keys_device,
            output_keys_device,
            num_elements,
            self._radix_sort_compute_flags_descending_function)

    def radix_sort_key_ascending_host(self, input_keys_host, output_keys_host):

        num_elements = input_keys_host.shape[0]
        
        self.__copy_input_htod_key(input_keys_host)
        self.radix_sort_key_ascending_device(self._input_keys_device, self._output_keys_device, num_elements)
        self.__copy_output_dtoh_key(output_keys_host)

    def radix_sort_key_descending_host(self, input_keys_host, output_keys_host):

        num_elements = input_keys_host.shape[0]
        
        self.__copy_input_htod_key(input_keys_host)
        self.radix_sort_key_descending_device(self._input_keys_device, self._output_keys_device, num_elements)
        self.__copy_output_dtoh_key(output_keys_host)

    def radix_sort_key_value_ascending_device(self, input_keys_device, input_values_device, output_keys_device, output_values_device, num_elements):

        self.__radix_sort_key_value(
            input_keys_device,
            input_values_device,
            output_keys_device,
            output_values_device,
            num_elements,
            self._radix_sort_compute_flags_ascending_function)

    def radix_sort_key_value_descending_device(self, input_keys_device, input_values_device, output_keys_device, output_values_device, num_elements):

        self.__radix_sort_key_value(
            input_keys_device,
            input_values_device,
            output_keys_device,
            output_values_device,
            num_elements,
            self._radix_sort_compute_flags_descending_function)

    def radix_sort_key_value_ascending_host(self, input_keys_host, input_values_host, output_keys_host, output_values_host):

        num_elements = input_keys_host.shape[0]
        
        self.__copy_input_htod_key_value(input_keys_host, input_values_host)
        self.radix_sort_key_value_ascending_device(self._input_keys_device, self._input_values_device, self._output_keys_device, self._output_values_device, num_elements)
        self.__copy_output_dtoh_key_value(output_keys_host, output_values_host)

    def radix_sort_key_value_descending_host(self, input_keys_host, input_values_host, output_keys_host, output_values_host):

        num_elements = input_keys_host.shape[0]
        
        self.__copy_input_htod_key_value(input_keys_host, input_values_host)
        self.radix_sort_key_value_descending_device(self._input_keys_device, self._input_values_device, self._output_keys_device, self._output_values_device, num_elements)
        self.__copy_output_dtoh_key_value(output_keys_host, output_values_host)

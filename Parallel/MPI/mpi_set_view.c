#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 1000

#define CHECK_ERR(func){ \
    if( err != MPI_SUCCESS  ){ \
        int errStringLen; \
        char errString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errString, &errStringLen); \
        printf("Error at line %d: calling %s (%s) \n", __LINE__, #func, errString); \
    } \
}

int main(int argc, char** argv){
    int rank, size, err, amode, inints, i;
    int count;
    int buf_w[COUNT];
    int buf_r[COUNT];
    char* filename;

    MPI_Aint lb, extent;
    
    
    MPI_File fh;
    MPI_Comm comm;
    MPI_Status* status;
    MPI_Datatype contig, etype, filetype;
    MPI_Info info;
    MPI_Offset filesize, offset, disp;

    comm   = MPI_COMM_WORLD;
    info   = MPI_INFO_NULL;
    status = MPI_STATUS_IGNORE;
    etype  = MPI_INT;
    
    MPI_Init(&argc, &argv);
    err = MPI_Comm_rank(comm, &rank);
    CHECK_ERR( MPI_Comm_rank );
    err = MPI_Comm_size(comm, &size);
    CHECK_ERR( MPI_Comm_size );
    
    filename = "test_mpi_io.dat";
    if( argc > 1 ) filename = argv[1];

/* * 
 * open file collectively, each process get's the size of it, 
 * and prints it.
 */
    amode  = MPI_MODE_CREATE;
    amode |= MPI_MODE_WRONLY;
    err = MPI_File_open(comm, filename, amode, info, &fh);
    CHECK_ERR( MPI_File_open to write );
    
    err = MPI_File_get_size(fh, &filesize);
    CHECK_ERR( MPI_File_get_size );

    printf("Process %d: Gets filesize = %lld Bytes.\n", rank, filesize);

    MPI_File_close(&fh);

/**
 * Each process reads the file at a explicit offset
 */
    err = MPI_File_open(comm, filename, MPI_MODE_RDONLY, info, &fh);
    inints = filesize / (size * sizeof(int));
    offset = rank * inints * sizeof(int);
    MPI_File_read_at(fh, offset, buf_r, inints, etype, status);



    printf("Process %d: Reads count = %d Bytes.\n", rank, count);
    MPI_File_close(&fh);




program_exit:
    MPI_Finalize();
    return 0;


}

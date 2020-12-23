#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 1000000

/**
 * # sign is called stringizing operator (#), more info at:
 * https://docs.microsoft.com/en-us/cpp/preprocessor/stringizing-operator-hash
 */
#define CHECK_ERR(func){ \
    if( err != MPI_SUCCESS ){ \
        int errStringLen; \
        char errString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errString, &errStringLen); \
        printf("Error at line %d: calling %s (%s)\n", __LINE__, #func, errString); \
    } \
}

int main(int argc, char** argv){

    int amode, err, rank, size;
    int buf[COUNT];
    char* filename;
    MPI_File fh;
    MPI_Comm comm;
    MPI_Info info;
    MPI_Status *status;
    MPI_Datatype datatype;

    comm     = MPI_COMM_WORLD;
    info     = MPI_INFO_NULL;
    status   = MPI_STATUS_IGNORE;
    datatype = MPI_INT;

    MPI_Init(&argc, &argv);
    err = MPI_Comm_rank(comm, &rank);
    CHECK_ERR( MPI_Comm_rank );
    err = MPI_Comm_size(comm, &size);
    CHECK_ERR( MPI_Comm_size );

    filename = "test_mpi_io.dat";
    if( argc > 1 )
        filename = argv[1];

    amode  = MPI_MODE_CREATE;
    amode |= MPI_MODE_WRONLY;
    err = MPI_File_open(comm, filename, amode, info, &fh);
    CHECK_ERR( MPI_File_open to write );
    
    if(rank == 0)
        MPI_File_write(fh, buf, COUNT, datatype, status);


    err = MPI_File_close(&fh);
    CHECK_ERR( MPI_File_close );

program_exit:
    MPI_Finalize();
    return 0;
}

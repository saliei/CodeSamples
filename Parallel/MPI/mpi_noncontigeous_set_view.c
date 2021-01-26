
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 1000000

#define CHECK_ERR(func){ \
    if( err != MPI_SUCCESS  ){ \
        int errStringLen; \
        char errString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errString, &errStringLen); \
        printf("Error at line %d: calling %s (%s) \n", __LINE__, #func, errString); \
    } \
}

int main(int argc, char** argv){

    MPI_Aint lb, extent;
    MPI_Datatype contig, filetype;
    MPI_File fh;
    int size, rank, err;
    MPI_Comm comm = MPI_COMM_WORLD;

    lb = 0;
    extent = 6 * sizeof(int);

    MPI_Init(&argc, &argv);
    err = MPI_Comm_rank(comm, &rank);
    CHECK_ERR( MPI_Comm_rank );
    err = MPI_Comm_size(comm, &size);
    CHECK_ERR( MPI_Comm_size );
    MPI_Type_contiguous(2, MPI_INT, &contig);
    MPI_Type_create_resized(contig, lb, extent, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_open(MPI_COMM_WORLD, "test_mpi_io.dat", MPI_MODE_CREATE|MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    int disp = 5 * sizeof(int);
    MPI_File_set_view(fh, disp, MPI_INT, filetype, "native", MPI_INFO_NULL);
    int count = COUNT;
    int buf[count];
    MPI_File_write_all(fh, buf, count, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    MPI_Finalize();
    return 0;


}

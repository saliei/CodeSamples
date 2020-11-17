#include <stdlib.h>
#include <string.h>

#include "hash_table.h"

int main(int argc, char** argv)
{
    table* tb = new_table();
    del_table(tb);

    return 0;
}

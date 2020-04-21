typedef struct
{
    char* key;
    char* val;

} item;

typedef struct
{
    int size;
    int count;
    item** items;//an array of pointers, each elem is an item
} table;

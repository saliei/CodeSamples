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

table* new_table();
void del_table(table* t);
void insert(table* t, const char* k, const char* v);
char* search(table* t, const char* k);
void del_key(table* t, const char* k);

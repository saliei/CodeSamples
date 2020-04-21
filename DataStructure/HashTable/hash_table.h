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

static item* new_item(const char* k, const char* v);
static void del_item(item* i);
table* new_table();
void del_table(table* t);

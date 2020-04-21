#include <stdlib.h>
#include <string.h>

#include "hash_table.h"

// static fucntions are visible only in this file
// you can't use them in other files
static item* new_item(const char* k, const char* v)
{
   item* i = malloc(sizeof(item));
   i->key = strdup(k);
   i->val = strdup(v);
   return i;
}

static void del_item(item* i)
{
    free(i->key);
    free(i->val);
    free(i);
}

table* new_table()
{
    table* t = malloc(sizeof(table));
    t->size = 53;
    t->count = 0;
    t->items = calloc((size_t)t->size, sizeof(item*));
    return t;
}

void del_table(table* t)
{
    for(size_t l=0; l < t->size; ++l)
    {
        item* i = t->items[l];
        if(i != NULL) del_item(i);
    }
    free(t->items);
    free(t);
}



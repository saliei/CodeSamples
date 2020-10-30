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

static int hash(const char* s, const int a, const int m)
{
    long hash = 0;
    const int len_s = strlen(s);
    for(int i=0; i < len_s; ++i)
    {
        hash += (long)pow(a, len_s-(i+1)) * s[i];
        hash %= m;
    }

    return (int)hash;
}

static int get_hash(const char* s, const int num_buckets, const int attempt)
{
    const int hash_1 = hash(s, PRIME_1, num_buckets);
    const int hash_2 = hash(s, PRIME_2, num_buckets);
    
    return (hash_1 + (1 + hash_2) * attempt) % num_buckets;
}

void insert(table* t, const char* k, const char* v)
{
    item* i = new_item(k, v);
    int idx = get_hash(i->key, t->size, 0);
    item* curr_item = t->items[idx];
    i = 1;
    while(curr_item != NULL)
    {
        idx = get_hash(i->key, t->size, i);
        curr_item = t->items[idx];
        i++;
    }

    t->items[idx] = i;
    t->count++;
}

char* search(table* t, const char* k)
{
    int idx = get_hash(k, t->size, 0);
    item* i = t->items[idx];
    int i = 1;
    while()

}

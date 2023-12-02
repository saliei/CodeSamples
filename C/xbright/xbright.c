/*
 *  xbright.c
 *  author: Alex Kozadaev (2011-2021)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "build_host.h"

#define NAME "xbright"

#define ERROR(msg)      { fprintf(stderr, "ERROR: %s - %s\n", __func__, \
                            (msg)); exit(EXIT_FAILURE); } while(0)
#ifdef DEBUG
#define DBG(fmt, ...) fprintf(stderr, fmt "\n", __VA_ARGS__);
#else
#define DBG(fmt, ...)
#endif

#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define MAX(a, b)       ((a) > (b) ? (a) : (b))

static int bright_up(const char*);
static int bright_down(const char*);
static void bright_set(const char*);
static int get_current(void);
static void commit_change(int);
static void usage(void);

int
main(int argc, char **argv)
{
/*    if (geteuid()) {
        usage();
        ERROR("Must be root");
        return 1;
    }*/

    if (argc > 1) {
        switch(*argv[1]) {
            case '+': {int pv = bright_up(++argv[1]); printf("%d\n", pv); break;}
            case '-': {int pv = bright_down(++argv[1]); printf("%d\n", pv); break;}
            case '=': bright_set(++argv[1]); break;
            default: usage();
        }
    } else {
        usage();
    }
    return EXIT_SUCCESS;
}

int
bright_up(const char *value)
{
    int offset_value = MAX(atoi(value), 1);
    int cur_value = get_current();
    int new_value = MIN(cur_value + offset_value, MAXVALUE);
    int percent_value = (int)((new_value * 100) / MAXVALUE);

    if (new_value != cur_value) {
        commit_change(new_value);
    }

    return percent_value;
}

int
bright_down(const char *value)
{
    int offset_value = MAX(atoi(value), 1);
    int cur_value = get_current();
    int new_value = MAX(cur_value - offset_value, 0);
    int percent_value = (int)((new_value * 100) / MAXVALUE);

    if (new_value != cur_value) {
        commit_change(new_value);
    }

    return percent_value;
}

void
bright_set(const char *value)
{
    int set_value = MIN(atoi(value), MAXVALUE);
    commit_change(set_value);
}

int
get_current(void)
{
    int value = 0;
    FILE* input = fopen(BRIGHTNESSFILE, "r");
    if (input == NULL) {
        ERROR("Cannot open kernel pipe");
    }
    (void)fscanf(input, "%d", &value);
    return value;
}

void
commit_change(int value)
{
    FILE* output = fopen(BRIGHTNESSFILE, "w");
    if (output == NULL) {
        ERROR("Cannot open kernel pipe");
    }
    //DBG("new value: %d", value);
    fprintf(output, "%d", value);
    fclose(output);
}

void
usage(void)
{
    printf("%s v%s (works only with 2.6+ kernels).\n", NAME, VERSION);
    printf("usage: %s [+-=[0-%d]]\n", NAME, MAXVALUE);
}

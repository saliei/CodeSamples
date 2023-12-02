#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/XKBlib.h>
#include <X11/extensions/XKBrules.h>

int main(int argc, char **argv) {
  Display *dpy = XOpenDisplay(NULL);

  if (dpy == NULL) {
    fprintf(stderr, "Cannot open display\n");
    exit(1);
  }

  XkbStateRec state;
  XkbGetState(dpy, XkbUseCoreKbd, &state);

  XkbDescPtr desc = XkbGetKeyboard(dpy, XkbAllComponentsMask, XkbUseCoreKbd);
  char *group = XGetAtomName(dpy, desc->names->groups[state.group]);
  printf("Full name: %s\n", group);

  XkbRF_VarDefsRec vd;
  XkbRF_GetNamesProp(dpy, NULL, &vd);

  char *tok = strtok(vd.layout, ",");

  for (int i = 0; i < state.group; i++) {
    tok = strtok(NULL, ",");
    if (tok == NULL) {
      return 1;
    }
  }

  printf("Layout name: %s\n", tok);

  return 0;
}

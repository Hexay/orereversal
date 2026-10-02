// Locating and running harness/region_dump, the cubiomes-backed generator refine uses for the families
// the GPU doesn't generate.
#ifndef REGION_DUMP_H
#define REGION_DUMP_H
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "ore_config.h"

#ifdef _WIN32
#define popen      _popen
#define pclose     _pclose
#define DEVNULL    "NUL"
#define PATHSEP    "\\"
#define EXE_SUFFIX ".exe"
#else
#define DEVNULL    "/dev/null"
#define PATHSEP    "/"
#define EXE_SUFFIX ""
#endif
#define REGION_DUMP_RELPATH ".." PATHSEP "harness" PATHSEP "region_dump" EXE_SUFFIX

static char g_regionDumpPath[600] = REGION_DUMP_RELPATH;

// region_dump lives in ../harness relative to this binary, wherever it is run from.
static void resolveRegionDumpPath(const char* argv0) {
    int lastSep = -1;
    for (int i = 0; argv0[i]; i++)
        if (argv0[i] == '/' || argv0[i] == '\\')
            lastSep = i;
    if (lastSep >= 0)
        snprintf(g_regionDumpPath, sizeof(g_regionDumpPath), "%.*s" PATHSEP REGION_DUMP_RELPATH, lastSep,
                 argv0);
}

static bool regionDumpExists() {
    FILE* f = fopen(g_regionDumpPath, "rb");
    if (!f)
        return false;
    fclose(f);
    return true;
}

static int familyFromName(const char* name) {
    for (int f = 0; f < F_COUNT; f++)
        if (!strcmp(name, FAMILY_NAMES[f]))
            return f;
    return -1;
}

// Calls onBlock(family, x, y, z) for every deepslate-band candidate of `families` (space-separated
// names) in the chunk box. Returns false if region_dump couldn't be started.
template <class OnBlock>
static bool runRegionDump(uint64_t seed, const char* version, int chunkMinX, int chunkMaxX, int chunkMinZ,
                          int chunkMaxZ, const char* families, OnBlock onBlock) {
    char command[700];
    snprintf(command, sizeof(command), "\"%s\" %llu %s %d %d %d %d -64 -1 %s 2>" DEVNULL, g_regionDumpPath,
             (unsigned long long)seed, version, chunkMinX, chunkMaxX, chunkMinZ, chunkMaxZ, families);
    FILE* p = popen(command, "r");
    if (!p)
        return false;
    char line[128];
    while (fgets(line, sizeof(line), p)) {
        char name[32];
        int x, y, z;
        if (sscanf(line, "%31[^,],%d,%d,%d", name, &x, &y, &z) != 4 || y < -64 || y >= 0)
            continue;
        int family = familyFromName(name);
        if (family >= 0)
            onBlock(family, x, y, z);
    }
    pclose(p);
    return true;
}

#endif

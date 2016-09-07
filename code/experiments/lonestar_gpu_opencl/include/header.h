// A platform portable timer. Works on Windows and Linux. Returns the
// current time as a double and in seconds. Linux has more precision
// than Windows. The most precise windows gets is milliseconds, while
// Linux has nano-seconds.

// Part of the OpenCL (and cross platform) GPU-Lonestar port by Tyler
// Sorensen (2016)

#pragma once

#include <stdio.h>
#include <time.h>
#include <fstream>
#include <string>
#include <iostream>

#include <cassert>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#ifdef WIN32 // For Windows

#include <Windows.h>

#else // For Linux

#include <sys/time.h>

#endif

#include <stdlib.h>
#include <stdarg.h>

double rtclock() {

#ifdef WIN32 // For Windows

  SYSTEMTIME time;
  GetSystemTime(&time);
  double ret = (time.wHour * 60 * 60 * 1000) + (time.wMinute * 60 * 1000) + (time.wSecond * 1000) + time.wMilliseconds;
  return ret/1000;

#else // For Linux

  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;

#endif

}

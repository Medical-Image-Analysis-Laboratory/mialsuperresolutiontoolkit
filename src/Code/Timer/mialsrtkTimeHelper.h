/*==========================================================================

  © Université de Lausanne - Medical Image Analysis Laboratory

  Date: 06/06/2014
  Author(s): Sebastien Tourbier (sebastien.tourbier@unil.ch)

  The user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

==========================================================================*/

#ifndef MIALTK_TIME_HELPER_H
#define MIALTK_TIME_HELPER_H

/* Std includes*/
#include <string>
#include <stdlib.h>

/* Time profiling */
#ifdef __MACH__
#include <time.h>
#include <mach/clock.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
#else
    #if TIME_WITH_SYS_TIME
        #include <sys/time.h>
        #include <time.h>
    #else
        #if HAS_SYS_TIME
            #include <sys/time.h>
        #else
            #include <time.h>
        #endif
    #endif
#endif

namespace mialtk {


double getTime(void)
{
    struct timespec tv;

#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    if(clock_get_time(cclock, &mts) != 0) return 0;
    mach_port_deallocate(mach_task_self(), cclock);
    tv.tv_sec = mts.tv_sec;
    tv.tv_nsec = mts.tv_nsec;
#else
    #ifdef NDEBUG
            if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
    #else
            return 0;
    #endif
#endif
    return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}


void printTime(std::string algo, double time)
{
    std::cout << std::endl <<"##########################################################################################################" << std::endl;
#ifdef __MACH__
    std::cout << algo << " optimization took " << time << " (MAC time)" <<  std::endl;
#else
    std::cout << algo << " optimization took " << time << " (Unix time)" <<  std::endl;
#endif
    std::cout <<"##########################################################################################################" << std::endl << std::endl;

}

std::string getRealCurrentDate()
{
#ifdef __MACH__
    std::cout <<" Recently implemented on MAC" <<  std::endl;
    time_t rawtime;
    time (&rawtime);
    char* charDate = ctime (&rawtime);
    std::string strDate(charDate);
    strDate[strDate.length() - 1] = '\0';
    return strDate + std::string(" (MAC)");
#else
    time_t rawtime;
    time (&rawtime);
    char* charDate = ctime (&rawtime);
    std::string strDate(charDate);
    strDate[strDate.length() - 1] = '\0';
    return strDate;
#endif
}

} // namespace mialtk


#endif // MIALTK_TIME_HELPER_H

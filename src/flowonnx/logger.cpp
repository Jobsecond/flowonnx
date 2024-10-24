#include "logger.h"

#include <iostream>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <sstream>
#include <chrono>
#include <iomanip>

// #define WINDOWS_USE_ANSI_COLOR

#if defined(_WIN32) && !defined(WINDOWS_USE_ANSI_COLOR)
#define USE_WIN32_API_COLOR
#endif

#ifdef USE_WIN32_API_COLOR
#include <Windows.h>
#endif

#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"

#define WIN32_COLOR_BLACK         0
#define WIN32_COLOR_BLUE          1
#define WIN32_COLOR_GREEN         2
#define WIN32_COLOR_CYAN          3
#define WIN32_COLOR_RED           4
#define WIN32_COLOR_MAGENTA       5
#define WIN32_COLOR_YELLOW        6
#define WIN32_COLOR_WHITE         7
#define WIN32_COLOR_INTENSITY     8  // For brighter versions of colors


namespace flowonnx {
    static std::mutex logMutex;
    static LogLevel currentLevel = LogLevel_Debug;
    static Logger::Callback loggerCallback = Logger::printColorLog;

    static std::string currentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm;
#if defined(_WIN32) || defined(_WIN64)
        localtime_s(&now_tm, &now_c);
#else
        localtime_r(&now_c, &now_tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    static std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel_Fatal:
                return "FATAL";
            case LogLevel_Error:
                return "ERROR";
            case LogLevel_Warning:
                return "WARNING";
            case LogLevel_Info:
                return "INFO";
            case LogLevel_Debug:
                return "DEBUG";
            default:
                return "UNKNOWN";
        }
    }

#ifdef USE_WIN32_API_COLOR
    static HANDLE g_hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif

    void Logger::printColorLog(int level, const char *category, const char *message) {
#ifdef USE_WIN32_API_COLOR
        WORD colorWindows;
        switch (level) {
            case LogLevel_Fatal:
            case LogLevel_Error:
                colorWindows = WIN32_COLOR_RED | WIN32_COLOR_INTENSITY;
                break;
            case LogLevel_Warning:
                colorWindows = WIN32_COLOR_YELLOW | WIN32_COLOR_INTENSITY;
                break;
            case LogLevel_Info:
                colorWindows = WIN32_COLOR_GREEN;
                break;
            case LogLevel_Debug:
                colorWindows = WIN32_COLOR_CYAN;
                break;
            default:
                colorWindows = WIN32_COLOR_WHITE;
                break;
        }
#else
        const char *colorAnsi;
        // Choose color based on log level
        switch (level) {
            case LogLevel_Fatal:
            case LogLevel_Error:
                colorAnsi = COLOR_RED;
                break;
            case LogLevel_Warning:
                colorAnsi = COLOR_YELLOW;
                break;
            case LogLevel_Info:
                colorAnsi = COLOR_GREEN;
                break;
            case LogLevel_Debug:
                colorAnsi = COLOR_CYAN;
                break;
            default:
                colorAnsi = COLOR_RESET;
                break;
        }
#endif
        // Log level with color (if enabled)
        std::ostream &outputStream = std::cout;
#ifdef USE_WIN32_API_COLOR
        SetConsoleTextAttribute(g_hConsole, colorWindows);
#else
        outputStream << colorAnsi;
#endif
        outputStream << "[" << currentTimestamp() << "] "
                     << "[" << category << "] "
                     << "[" << levelToString(static_cast<LogLevel>(level)) << "]"
                     << " " << message;
#ifdef USE_WIN32_API_COLOR
        SetConsoleTextAttribute(g_hConsole, WIN32_COLOR_WHITE);
#else
        outputStream << COLOR_RESET;
#endif
        outputStream << std::endl;
    }


// Logger class definitions
    Logger::Logger() = default;

    Logger::~Logger() = default;

    void Logger::setLogLevel(LogLevel level) {
        std::lock_guard<std::mutex> guard(logMutex);
        currentLevel = level;
    }

    void Logger::setCallback(Callback callback) {
        loggerCallback = callback;
    }

    void Logger::setDefaultCallback() {
        loggerCallback = Logger::printColorLog;
    }

    void Logger::log(LogLevel level, const std::string &category, const std::string &message) {
        std::lock_guard<std::mutex> guard(logMutex);
        if (level <= currentLevel) {
            loggerCallback(level, category.c_str(), message.c_str());
        }
    }

    void Logger::fatal(const std::string &category, const std::string &message) {
        log(LogLevel_Fatal, category, message);
    }

    void Logger::error(const std::string &category, const std::string &message) {
        log(LogLevel_Error, category, message);
    }

    void Logger::warning(const std::string &category, const std::string &message) {
        log(LogLevel_Warning, category, message);
    }

    void Logger::info(const std::string &category, const std::string &message) {
        log(LogLevel_Info, category, message);
    }

    void Logger::debug(const std::string &category, const std::string &message) {
        log(LogLevel_Debug, category, message);
    }
}
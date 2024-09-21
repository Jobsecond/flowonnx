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

#ifdef USE_WIN32_API_COLOR
    static HANDLE g_hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#endif

    class Logger::Impl {
    public:
        struct LogEntry {
            LogLevel level;
            std::string timestamp;
            std::string message;
        };

        explicit Impl(LogLevel level) : currentLevel(level),
                                        timestampEnabled(true),
                                        consoleEnabled(true),
                                        logToStdErr(false),
                                        colorEnabled(true),
                                        exitThread(false),
                                        loggingThread(&Impl::loggingThreadFunc, this) {
        }

        ~Impl() {
            {
                std::lock_guard<std::mutex> guard(logMutex);
                exitThread = true;
            }
            logCondition.notify_one();
            loggingThread.join();
            if (logFile.is_open()) {
                logFile.close();
            }
        }

        void setLogLevel(LogLevel level) {
            std::lock_guard<std::mutex> guard(logMutex);
            currentLevel = level;
        }

        void enableTimestamp(bool enable) {
            std::lock_guard<std::mutex> guard(logMutex);
            timestampEnabled = enable;
        }

        void enableConsole(bool enable, bool useStdErr) {
            std::lock_guard<std::mutex> guard(logMutex);
            consoleEnabled = enable;
            logToStdErr = useStdErr;
        }

        void enableColor(bool enable) {
            std::lock_guard<std::mutex> guard(logMutex);
            colorEnabled = enable;
        }

        bool setLogFile(const std::string &filename) {
            std::lock_guard<std::mutex> guard(logMutex);
            if (logFile.is_open()) {
                logFile.close();
            }
            logFile.open(filename, std::ios::out | std::ios::app);
            return logFile.is_open();
        }

        void disableLogFile() {
            std::lock_guard<std::mutex> guard(logMutex);
            if (logFile.is_open()) {
                logFile.close();
            }
        }

        void log(LogLevel level, const std::string &message) {
            std::lock_guard<std::mutex> guard(logMutex);
            if (level <= currentLevel) {
                LogEntry entry{level, currentTimestamp(), message};
                logQueue.push(entry);
                logCondition.notify_one();
            }
        }

        void critical(const std::string &message) {
            log(LogLevel_Critical, message);
        }

        void error(const std::string &message) {
            log(LogLevel_Error, message);
        }

        void warning(const std::string &message) {
            log(LogLevel_Warning, message);
        }

        void info(const std::string &message) {
            log(LogLevel_Info, message);
        }

        void debug(const std::string &message) {
            log(LogLevel_Debug, message);
        }

    private:
        void loggingThreadFunc() {
            while (true) {
                std::unique_lock<std::mutex> lock(logMutex);
                logCondition.wait(lock, [this] { return !logQueue.empty() || exitThread; });

                while (!logQueue.empty()) {
                    LogEntry logEntry = logQueue.front();
                    logQueue.pop();

#ifdef USE_WIN32_API_COLOR
                    WORD colorWindows;
                    if (colorEnabled) {
                        switch (logEntry.level) {
                            case LogLevel_Critical:
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
                    }
#else
                    const char *colorAnsi;
                    if (colorEnabled) {
                        // Choose color based on log level
                        switch (logEntry.level) {
                            case LogLevel_Critical:
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
                    }
#endif

                    // Format log output
                    std::ostringstream formattedMessage;
                    if (timestampEnabled) {
                        formattedMessage << "[" << logEntry.timestamp << "] ";
                    }

                    // Log level with color (if enabled)
                    if (consoleEnabled) {
                        std::ostream &outputStream = logToStdErr ? std::cerr : std::cout;
#ifdef USE_WIN32_API_COLOR
                        SetConsoleTextAttribute(g_hConsole, colorWindows);
#else
                        outputStream << colorAnsi;
#endif
                        outputStream << "[" << logEntry.timestamp << "] "
                                     << "[" << levelToString(logEntry.level) << "]"
                                     << " " << logEntry.message;
#ifdef USE_WIN32_API_COLOR
                        SetConsoleTextAttribute(g_hConsole, WIN32_COLOR_WHITE);
#else
                        outputStream << COLOR_RESET;
#endif
                        outputStream << std::endl;
                    }

                    if (logFile.is_open()) {
                        logFile << "[" << logEntry.timestamp << "] "
                                << "[" << levelToString(logEntry.level) << "] "
                                << logEntry.message << std::endl;
                    }
                }

                if (exitThread && logQueue.empty()) {
                    break;
                }
            }
        }

        static std::string levelToString(LogLevel level) {
            switch (level) {
                case LogLevel_Critical:
                    return "CRITICAL";
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

        LogLevel currentLevel;
        std::mutex logMutex;
        std::ofstream logFile;
        bool timestampEnabled;
        bool consoleEnabled;
        bool logToStdErr;
        bool colorEnabled;

        std::queue<LogEntry> logQueue;
        std::thread loggingThread;
        std::condition_variable logCondition;
        std::atomic<bool> exitThread;
    };

// Logger class definitions
    Logger::Logger(LogLevel level) : _impl(std::make_unique<Impl>(level)) {}

    Logger::~Logger() = default;

    Logger &Logger::getInstance() {
        static Logger instance;
        return instance;
    }

    void Logger::setLogLevel(LogLevel level) {
        _impl->setLogLevel(level);
    }

    void Logger::enableTimestamp(bool enable) {
        _impl->enableTimestamp(enable);
    }

    void Logger::enableConsole(bool enable, bool useStdErr) {
        _impl->enableConsole(enable, useStdErr);
    }

    void Logger::enableColor(bool enable) {
        _impl->enableColor(enable);
    }

    bool Logger::setLogFile(const std::string &filename) {
        return _impl->setLogFile(filename);
    }

    void Logger::disableLogFile() {
        _impl->disableLogFile();
    }

    void Logger::log(LogLevel level, const std::string &message) {
        _impl->log(level, message);
    }

    void Logger::critical(const std::string &message) {
        _impl->critical(message);
    }

    void Logger::error(const std::string &message) {
        _impl->error(message);
    }

    void Logger::warning(const std::string &message) {
        _impl->warning(message);
    }

    void Logger::info(const std::string &message) {
        _impl->info(message);
    }

    void Logger::debug(const std::string &message) {
        _impl->debug(message);
    }
}
#include "Logger.h"
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

    class Logger::Impl {
    public:
        explicit Impl(LogLevel level) : currentLevel(level), timestampEnabled(true), consoleEnabled(true), exitThread(false),
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

        void enableConsole(bool enable) {
            std::lock_guard<std::mutex> guard(logMutex);
            consoleEnabled = enable;
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
                std::ostringstream oss;
                if (timestampEnabled) {
                    oss << "[" << currentTimestamp() << "] ";
                }
                oss << "[" << levelToString(level) << "] " << message;

                logQueue.push(oss.str());
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
                    std::string logMessage = logQueue.front();
                    logQueue.pop();

                    if (consoleEnabled) {
                        std::cout << logMessage << std::endl;
                    }

                    if (logFile.is_open()) {
                        logFile << logMessage << std::endl;
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

        std::queue<std::string> logQueue;
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

    void Logger::enableConsole(bool enable) {
        _impl->enableConsole(enable);
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
#ifndef LOGGER_H
#define LOGGER_H

#include <memory>
#include <string>

#include <flowonnx/flowonnxglobal.h>
#include <flowonnx/format.h>

namespace flowonnx {

    enum LogLevel {
        LogLevel_Off = 0,
        LogLevel_Fatal = 1,
        LogLevel_Error = 2,
        LogLevel_Warning = 3,
        LogLevel_Info = 4,
        LogLevel_Debug = 5,
    };

    class FLOWONNX_EXPORT Logger {
    public:
        Logger();

        ~Logger();

        Logger(const Logger &) = delete;

        Logger &operator=(const Logger &) = delete;

        using Callback = void (*)(int, const char *, const char *);
    public:
        static void printColorLog(int level, const char *category, const char *message);

        static void setLogLevel(LogLevel level);

        static void setCallback(Callback callback);

        static void setDefaultCallback();

        static void log(LogLevel level, const std::string &category, const std::string &message);

        template <typename... Args>
        static void log(LogLevel level, const std::string &category, const std::string &format, Args &&...args) {
            log(level, category, formatTextN(format, std::forward<Args>(args)...));
        }

        static void fatal(const std::string &category, const std::string &message);

        template <typename... Args>
        static void fatal(const std::string &category, const std::string &format, Args &&...args) {
            fatal(category, formatTextN(format, std::forward<Args>(args)...));
        }

        static void error(const std::string &category, const std::string &message);

        template <typename... Args>
        static void error(const std::string &category, const std::string &format, Args &&...args) {
            error(category, formatTextN(format, std::forward<Args>(args)...));
        }

        static void warning(const std::string &category, const std::string &message);

        template <typename... Args>
        static void warning(const std::string &category, const std::string &format, Args &&...args) {
            warning(category, formatTextN(format, std::forward<Args>(args)...));
        }

        static void info(const std::string &category, const std::string &message);

        template <typename... Args>
        static void info(const std::string &category, const std::string &format, Args &&...args) {
            info(category, formatTextN(format, std::forward<Args>(args)...));
        }

        static void debug(const std::string &category, const std::string &message);

        template <typename... Args>
        static void debug(const std::string &category, const std::string &format, Args &&...args) {
            debug(category, formatTextN(format, std::forward<Args>(args)...));
        }
    };

}

#define LOG_WITH_LEVEL(level, category, format, ...) \
    flowonnx::Logger::log(level, category, format, ##__VA_ARGS__)

#define LOG_FATAL(category, format, ...) \
    flowonnx::Logger::fatal(category, format, ##__VA_ARGS__)

#define LOG_ERROR(category, format, ...) \
    flowonnx::Logger::error(category, format, ##__VA_ARGS__)

#define LOG_WARNING(category, format, ...) \
    flowonnx::Logger::warning(category, format, ##__VA_ARGS__)

#define LOG_INFO(category, format, ...) \
    flowonnx::Logger::info(category, format, ##__VA_ARGS__)

#define LOG_DEBUG(category, format, ...) \
    flowonnx::Logger::debug(category, format, ##__VA_ARGS__)

#endif // LOGGER_H
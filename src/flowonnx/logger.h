#ifndef LOGGER_H
#define LOGGER_H

#include <memory>
#include <string>

#include <flowonnx/flowonnxglobal.h>
#include <flowonnx/format.h>

namespace flowonnx {

    enum LogLevel {
        LogLevel_Off = 0,
        LogLevel_Critical = 1,
        LogLevel_Error = 2,
        LogLevel_Warning = 3,
        LogLevel_Info = 4,
        LogLevel_Debug = 5,
    };

    class FLOWONNX_EXPORT Logger {
    public:
        static Logger &getInstance();

        void setLogLevel(LogLevel level);

        void enableTimestamp(bool enable = true);

        void enableConsole(bool enable = true, bool useStdErr = false);

        void enableColor(bool enable = true);

        bool setLogFile(const std::string &filename); // Returns true if successful, false otherwise

        void disableLogFile();

        void log(LogLevel level, const std::string &message);

        template <typename... Args>
        void log(LogLevel level, const std::string &format, Args &&...args) {
            log(level, formatTextN(format, std::forward<Args>(args)...));
        }

        void critical(const std::string &message);

        template <typename... Args>
        void critical(const std::string &format, Args &&...args) {
            critical(formatTextN(format, std::forward<Args>(args)...));
        }

        void error(const std::string &message);

        template <typename... Args>
        void error(const std::string &format, Args &&...args) {
            error(formatTextN(format, std::forward<Args>(args)...));
        }

        void warning(const std::string &message);

        template <typename... Args>
        void warning(const std::string &format, Args &&...args) {
            warning(formatTextN(format, std::forward<Args>(args)...));
        }

        void info(const std::string &message);

        template <typename... Args>
        void info(const std::string &format, Args &&...args) {
            info(formatTextN(format, std::forward<Args>(args)...));
        }

        void debug(const std::string &message);

        template <typename... Args>
        void debug(const std::string &format, Args &&...args) {
            debug(formatTextN(format, std::forward<Args>(args)...));
        }

    private:
        explicit Logger(LogLevel level = LogLevel_Debug);

        ~Logger();

        Logger(const Logger &) = delete;

        Logger &operator=(const Logger &) = delete;

        class Impl;

        std::unique_ptr<Impl> _impl;
    };

}

#define FLOWONNX_LOG(level, format, ...) \
    flowonnx::Logger::getInstance().log(level, format, ##__VA_ARGS__)

#define FLOWONNX_CRITICAL(format, ...) \
    flowonnx::Logger::getInstance().critical(format, ##__VA_ARGS__)

#define FLOWONNX_ERROR(format, ...) \
    flowonnx::Logger::getInstance().error(format, ##__VA_ARGS__)

#define FLOWONNX_WARNING(format, ...) \
    flowonnx::Logger::getInstance().warning(format, ##__VA_ARGS__)

#define FLOWONNX_INFO(format, ...) \
    flowonnx::Logger::getInstance().info(format, ##__VA_ARGS__)

#define FLOWONNX_DEBUG(format, ...) \
    flowonnx::Logger::getInstance().debug(format, ##__VA_ARGS__)

#endif // LOGGER_H
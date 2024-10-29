#ifndef FLOWONNX_SESSIONSYSTEM_P_H
#define FLOWONNX_SESSIONSYSTEM_P_H

#include <map>
#include <filesystem>
#include <shared_mutex>

namespace flowonnx {

class SessionImage;

class SessionSystem {
private:
    friend class Environment;
    SessionSystem();
public:
    SessionSystem(const SessionSystem &) = delete;
    SessionSystem &operator=(const SessionSystem &) = delete;

    ~SessionSystem();

    static SessionSystem *instance();

    bool addImage(const std::filesystem::path &path, SessionImage *image, bool overwrite = false);
    bool addImage(std::filesystem::path &&path, SessionImage *image, bool overwrite = false);
    bool removeImage(const std::filesystem::path &path);
    SessionImage *getImage(const std::filesystem::path &path);
private:
    mutable std::shared_mutex mtx;
    std::map<std::filesystem::path, SessionImage *> sessionImageMap;

};

}

#endif // FLOWONNX_SESSIONSYSTEM_P_H
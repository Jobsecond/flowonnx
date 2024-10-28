#ifndef FLOWONNX_SESSIONSYSTEM_P_H
#define FLOWONNX_SESSIONSYSTEM_P_H

#include <map>
#include <filesystem>

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

    std::map<std::filesystem::path, SessionImage *> sessionImageMap;

    static SessionSystem *instance();
};

}

#endif // FLOWONNX_SESSIONSYSTEM_P_H
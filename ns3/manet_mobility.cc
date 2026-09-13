#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace ns3;

static NodeContainer g_nodes;
static std::ofstream g_out;
static double g_interval = 10.0;
static double g_stop = 500.0;
static uint32_t g_run = 1;

static void
WriteSnapshot()
{
    const double now = Simulator::Now().GetSeconds();

    for (uint32_t i = 0; i < g_nodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mob =
            g_nodes.Get(i)->GetObject<MobilityModel>();

        Vector p = mob->GetPosition();
        Vector v = mob->GetVelocity();

        double speed =
            std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

        g_out << g_run << ','
              << std::fixed << std::setprecision(2)
              << now << ','
              << i << ','
              << p.x << ','
              << p.y << ','
              << speed << '\n';
    }

    if (now + g_interval <= g_stop + 1e-9)
    {
        Simulator::Schedule(
            Seconds(g_interval),
            &WriteSnapshot);
    }
}

int
main(int argc, char* argv[])
{
    uint32_t numNodes = 100;
    double simTime = 500.0;
    double snapshotInterval = 10.0;

    double minSpeed = 5.0;
    double maxSpeed = 25.0;
    double pause = 10.0;

    double areaX = 1000.0;
    double areaY = 1000.0;

    uint32_t runId = 1;

    std::string output = "positions_run1.csv";

    CommandLine cmd(__FILE__);

    cmd.AddValue(
        "nodes",
        "Number of mobile MANET nodes",
        numNodes);

    cmd.AddValue(
        "simTime",
        "Simulation time in seconds",
        simTime);

    cmd.AddValue(
        "interval",
        "Snapshot interval in seconds",
        snapshotInterval);

    cmd.AddValue(
        "minSpeed",
        "Minimum Random Waypoint speed (m/s)",
        minSpeed);

    cmd.AddValue(
        "maxSpeed",
        "Maximum Random Waypoint speed (m/s)",
        maxSpeed);

    cmd.AddValue(
        "pause",
        "Random Waypoint pause time (s)",
        pause);

    cmd.AddValue(
        "areaX",
        "Simulation width (m)",
        areaX);

    cmd.AddValue(
        "areaY",
        "Simulation height (m)",
        areaY);

    cmd.AddValue(
        "run",
        "ns-3 RNG run number",
        runId);

    cmd.AddValue(
        "output",
        "CSV output path",
        output);

    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(runId);

    g_nodes.Create(numNodes);

    ObjectFactory pos;
    pos.SetTypeId(
        "ns3::RandomRectanglePositionAllocator");

    std::ostringstream xRv, yRv;

    xRv << "ns3::UniformRandomVariable[Min=0.0|Max="
        << areaX << "]";

    yRv << "ns3::UniformRandomVariable[Min=0.0|Max="
        << areaY << "]";

    pos.Set(
        "X",
        StringValue(xRv.str()));

    pos.Set(
        "Y",
        StringValue(yRv.str()));

    Ptr<PositionAllocator> allocator =
        pos.Create()->GetObject<PositionAllocator>();

    std::ostringstream speedRv, pauseRv;

    speedRv
        << "ns3::UniformRandomVariable[Min="
        << minSpeed
        << "|Max="
        << maxSpeed
        << "]";

    pauseRv
        << "ns3::ConstantRandomVariable[Constant="
        << pause
        << "]";

    MobilityHelper mobility;

    mobility.SetMobilityModel(
        "ns3::RandomWaypointMobilityModel",
        "Speed",
        StringValue(speedRv.str()),
        "Pause",
        StringValue(pauseRv.str()),
        "PositionAllocator",
        PointerValue(allocator));

    mobility.SetPositionAllocator(allocator);

    mobility.Install(g_nodes);

    mobility.AssignStreams(g_nodes, 0);

    g_out.open(output, std::ios::out);

    if (!g_out.is_open())
    {
        NS_FATAL_ERROR(
            "Could not open output file: "
            << output);
    }

    g_out << "run,time,node_id,x,y,speed\n";

    g_interval = snapshotInterval;
    g_stop = simTime;
    g_run = runId;

    Simulator::Schedule(
        Seconds(0.0),
        &WriteSnapshot);

    Simulator::Stop(
        Seconds(simTime + 0.001));

    Simulator::Run();
    Simulator::Destroy();

    g_out.close();

    std::cout
        << "Created "
        << output
        << " for "
        << numNodes
        << " nodes, run "
        << runId
        << std::endl;

    return 0;
}
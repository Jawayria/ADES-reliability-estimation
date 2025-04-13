def create_system_elements():
    global slaves
    global switches
    global links

    global slave_to_link_edges
    global switch_to_link_edges

    slaves = tuple(Slave() for i in range(NUM_SLAVES))
    switches = tuple(Switch() for i in range(NUM_SWITCHES))

    for slave in slaves:
        if slave.index == 1:
            slink = Slink(str(slave.index) + str(3))
            links.append(slink)

            slave_to_link_edges.append((slave, slink))
            switch_to_link_edges.append((switches[2], slink))
        if slave.index == 2:
            slink = Slink(str(slave.index) + str(2))
            links.append(slink)

            slave_to_link_edges.append((slave, slink))
            switch_to_link_edges.append((switches[1], slink))
        if slave.index == 3:
            slink = Slink(str(slave.index) + str(1))
            links.append(slink)

            slave_to_link_edges.append((slave, slink))
            switch_to_link_edges.append((switches[0], slink))

            slink = Slink(str(slave.index) + str(3))
            links.append(slink)

            slave_to_link_edges.append((slave, slink))
            switch_to_link_edges.append((switches[2], slink))
        if slave.index == 4:
            slink = Slink(str(slave.index) + str(1))
            links.append(slink)

            slave_to_link_edges.append((slave, slink))
            switch_to_link_edges.append((switches[0], slink))

            slink = Slink(str(slave.index) + str(3))
            links.append(slink)

            slave_to_link_edges.append((slave, slink))
            switch_to_link_edges.append((switches[2], slink))
        if slave.index == 5:
            for switch in switches:
                slink = Slink(str(slave.index) + str(1))
                links.append(slink)

                slave_to_link_edges.append((slave, slink))
                switch_to_link_edges.append((switches[0], slink))

                slink = Slink(str(slave.index) + str(2))
                links.append(slink)

                slave_to_link_edges.append((slave, slink))
                switch_to_link_edges.append((switches[1], slink))

    # Create clique of switches interconnected by interlinks
    i = 0
    for switch1, switch2 in combinations(switches, 2):
        for j in range(NUM_ILINKS):
            ilink = Ilink(str(switch1.index) + str(switch2.index) + str(j + 1))
            links.append(ilink)

            switch_to_link_edges.append((switch1, ilink))
            switch_to_link_edges.append((switch2, ilink))
        i += NUM_ILINKS

    links = tuple(links)

    failure_rate_vector.update({s: FR_SLAVES for s in slaves})
    failure_rate_vector.update({l: FR_LINKS for l in links})
    failure_rate_vector.update({b: FR_SWITCHES for b in switches})

    print("SYSTEM")
    print("Number of switches: " + str(NUM_SWITCHES))
    print("Number of slaves: " + str(NUM_SLAVES))
    print("Number of req slaves: " + str(NUM_REQ_SLAVES))
    print("")

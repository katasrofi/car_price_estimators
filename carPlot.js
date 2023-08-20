export const animatedPlot = () => {
    let width;
    let height;
    let data;
    let x_value;
    let y_value;
    let margin;
    let radius;

    const my = (svg) => {
        // Adjust coordinat x and y
    const x = d3.scaleLinear()
        .domain(d3.extent(data, x_value))
        .range([margin.left, width - margin.right]);
    const y = d3.scaleLinear()
        .domain(d3.extent(data, y_value))
        .range([height - margin.bottom, margin.top]);

    const marks = data.map(d => ({
        x: x(x_value(d)),
        y: y(y_value(d)),
    }));

    const t = d3.transition()
        .duration(1050);

    const positionCircles = circles => {
        circles
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
    }
    const circles = svg
        .selectAll('circle')
        .data(marks)
        .join(
            enter => enter.append('circle')
                    .call(positionCircles)
                    .attr('r', 0)
                    .call(enter => 
                        enter
                        .transition(t)
                        .attr('r', 5)
                        ),
            update => update
                    .call(update => 
                        update
                        .transition(t)
                        .delay((d, i) => i * 10)
                        .call(positionCircles)
                        ),
            exit => exit.remove()
        );

    svg.selectAll('.y-axis')
        .data([null])
        .join('g')
        .attr('class', 'y-axis')
        .attr('transform', 
            `translate(${margin.left}, 0)`
            )
        .transition(t)
        .call(d3.axisLeft(y));

    svg.selectAll('.x-axis')
        .data([null])
        .join('g')
        .attr('class', 'x-axis')
        .attr('transform', 
            `translate(0, ${height-margin.bottom})`
            )
        .transition(t)
        .call(d3.axisBottom(x));
    };

    my.width = function(_) {
        return arguments.length ? ((width = +_), my) : width;
    };

    my.height = function(_) {
        return arguments.length ? ((height = +_), my) : height;
    };

    my.data = function(_) {
        return arguments.length ? ((data = _), my) : data;
    };

    my.x_value = function(_) {
        return arguments.length ? ((x_value = _), my) : x_value;
    };

    my.y_value = function(_) {
        return arguments.length ? ((y_value = _), my) : y_value;
    };

    my.margin = function(_) {
        return arguments.length ? ((margin = _), my) : margin;
    };

    my.radius = function(_) {
        return arguments.length ? ((radius = +_), my) : radius;
    };

    return my;
};

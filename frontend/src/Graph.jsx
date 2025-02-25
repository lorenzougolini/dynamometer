import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const Graph = ({data}) => {
    if (data.length === 0) return null;

    return (
        <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="force" stroke="#8884d8" />
            </LineChart>
        </ResponsiveContainer>
    )
}

export default Graph
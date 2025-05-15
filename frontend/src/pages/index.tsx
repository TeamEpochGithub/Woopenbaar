import type { NextPage } from 'next';
import { Layout } from '@/components/Layout';
import { ChatWindow } from '@/components/chat/ChatWindow';

const Home: NextPage = () => {
  return (
    <Layout>
      <div className="py-8">
        <ChatWindow />
      </div>
    </Layout>
  );
};

export default Home; 